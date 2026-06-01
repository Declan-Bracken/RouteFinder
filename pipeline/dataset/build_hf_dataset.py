"""
Build a HuggingFace dataset from CLIP-filtered MP route images.

Queries route_image_filter_scores for images that passed the filter at the given
model version and threshold, joins back full metadata from the manifest CSV,
downloads and resizes each image, and pushes to HuggingFace in shards.

Run apply_clip_filter.py first, inspect the score distribution, then call this
with your chosen threshold. The threshold and model_version are stored as columns
in the dataset for provenance.

Usage:
    python scripts/build_hf_dataset.py \\
        --hf_repo DeclanBracken/RouteFinderDatasetV3 \\
        --model_version best \\
        --threshold 0.90
"""

import argparse
import asyncio
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp
import pandas as pd
import psycopg2
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

import tempfile
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import HfApi

load_dotenv()

MANIFEST = Path(__file__).parent.parent / "data" / "mp_dataset_manifest.csv"
HEADERS  = {"User-Agent": "Mozilla/5.0 (compatible; RouteFinder-dataset/1.0)"}
RESIZE_PX = 448


# ── Download ──────────────────────────────────────────────────────────────────

async def _fetch(session, url, sem):
    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=20)) as r:
                    if r.status == 200:
                        return await r.read()
                    if r.status in (403, 404, 410):
                        return None
            except Exception:
                pass
            await asyncio.sleep(2 ** attempt)
    return None


async def download_batch(urls, concurrency):
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrency)) as session:
        return await asyncio.gather(*[_fetch(session, url, sem) for url in urls])


# ── HF push ───────────────────────────────────────────────────────────────────

def ensure_repo(hf_repo, hf_token):
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(hf_repo, repo_type="dataset", exist_ok=True, private=True)


_FEATURES = Features({
    "image":                HFImage(),
    "route_image_id":       Value("int32"),
    "route_id":             Value("int32"),
    "route_name":           Value("string"),
    "grade":                Value("string"),
    "type":                 Value("string"),
    "description":          Value("string"),
    "location":             Value("string"),
    "gps":                  Value("string"),
    "area_id":              Value("int32"),
    "area_name":            Value("string"),
    "area_path":            Value("string"),
    "area_depth":           Value("int32"),
    "filter_score":         Value("float32"),
    "filter_model_version": Value("string"),
})


def push_shard(records, hf_repo, hf_token, shard_idx):
    api = HfApi(token=hf_token)
    shard = Dataset.from_list(records, features=_FEATURES)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        shard.to_parquet(f.name)
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo=f"data/train-{shard_idx:05d}.parquet",
            repo_id=hf_repo,
            repo_type="dataset",
            token=hf_token,
        )
    print(f"  Pushed shard {shard_idx:05d}: {len(records)} images")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo",       required=True,
                        help="HuggingFace dataset repo, e.g. DeclanBracken/RouteFinderDatasetV3")
    parser.add_argument("--model_version", required=True,
                        help="filter_model_version to pull from route_image_filter_scores")
    parser.add_argument("--threshold",     type=float, default=0.50)
    parser.add_argument("--manifest",      default=str(MANIFEST))
    parser.add_argument("--database_url",  default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--hf_token",      default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--shard_size",    type=int, default=5000,
                        help="Images per HF push (default: 5000)")
    parser.add_argument("--concurrency",   type=int, default=30)
    parser.add_argument("--resize_px",     type=int, default=RESIZE_PX,
                        help=f"Thumbnail max side before storing in parquet (default: {RESIZE_PX})")
    args = parser.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL not set — add it to .env or pass --database_url")
    if not args.hf_token:
        sys.exit("HF_TOKEN not set — add it to .env or pass --hf_token")

    # ── Pull passing image IDs + scores from DB ───────────────────────────────
    conn = psycopg2.connect(args.database_url)
    scores_df = pd.read_sql(
        """
        SELECT route_image_id, raw_score
        FROM route_image_filter_scores
        WHERE filter_model_version = %s
          AND raw_score >= %s
        """,
        conn,
        params=(args.model_version, args.threshold),
    )
    conn.close()
    print(f"Passing images in DB: {len(scores_df)}")

    if scores_df.empty:
        sys.exit("No passing images found — run apply_clip_filter.py first.")

    # ── Join with manifest for full metadata ──────────────────────────────────
    manifest = pd.read_csv(args.manifest)
    df = manifest.merge(scores_df, on="route_image_id", how="inner")

    # De-duplicate: keep one row per route_image_id (highest score wins)
    df = (df.sort_values("raw_score", ascending=False)
            .drop_duplicates(subset="route_image_id")
            .reset_index(drop=True))

    # Keep only routes with at least 2 passing images (positive pairs required for SupCon)
    route_counts = df.groupby("route_id")["route_image_id"].transform("count")
    before = len(df)
    df = df[route_counts >= 2].reset_index(drop=True)
    print(f"Dropped {before - len(df)} single-image routes")
    print(f"After dedup + pair filter: {len(df)} images across {df['route_id'].nunique()} routes")

    ensure_repo(args.hf_repo, args.hf_token)

    buffer     = []
    shard_idx  = 0
    pushed     = 0
    skipped    = 0
    chunk_size = 256

    for chunk_start in tqdm(range(0, len(df), chunk_size), desc="Building dataset", unit="chunk"):
        chunk    = df.iloc[chunk_start: chunk_start + chunk_size]
        raw_data = asyncio.run(download_batch(chunk["image_url"].tolist(), args.concurrency))

        for (_, row), data in zip(chunk.iterrows(), raw_data):
            if data is None:
                skipped += 1
                continue
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.thumbnail((args.resize_px, args.resize_px), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
                img_bytes = buf.getvalue()
            except Exception:
                skipped += 1
                continue

            buffer.append({
                "image":                {"bytes": img_bytes, "path": None},
                "route_image_id":       int(row["route_image_id"]),
                "route_id":             int(row["route_id"]),
                "route_name":           str(row["route_name"] or ""),
                "grade":                str(row.get("grade") or ""),
                "type":                 str(row.get("type") or ""),
                "description":          str(row.get("description") or ""),
                "location":             str(row.get("location") or ""),
                "gps":                  str(row.get("gps") or ""),
                "area_id":              int(row["area_id"]),
                "area_name":            str(row["area_name"]),
                "area_path":            str(row["area_path"]),
                "area_depth":           int(row["area_depth"]),
                "filter_score":         float(row["raw_score"]),
                "filter_model_version": str(args.model_version),
            })

        if len(buffer) >= args.shard_size:
            push_shard(buffer, args.hf_repo, args.hf_token, shard_idx)
            pushed    += len(buffer)
            shard_idx += 1
            buffer     = []

    if buffer:
        push_shard(buffer, args.hf_repo, args.hf_token, shard_idx)
        pushed += len(buffer)

    print(f"\nDone. Pushed {pushed} images, skipped {skipped} failed downloads.")
    print(f"Dataset: https://huggingface.co/datasets/{args.hf_repo}")
    print(f"\nNext: update Config.hf_dataset in train/train.py to '{args.hf_repo}'")


if __name__ == "__main__":
    main()
