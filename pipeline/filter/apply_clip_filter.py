"""
Score all MP route images with the CLIP filter and log results to route_image_filter_scores.

Resumable — skips images already scored under the same model version. Run this first,
inspect the score distribution, then choose a threshold and run build_hf_dataset.py.

Prerequisites:
    - Run app/migrations/004_route_image_filter_scores.sql against the live DB
    - Have data/mp_dataset_manifest.csv (output of export_mp_dataset.py)
    - Have a CLIP filter checkpoint (e.g. models/clip_filter/best.pth)

Usage:
    python scripts/apply_clip_filter.py --checkpoint models/clip_filter/best.pth
    python scripts/apply_clip_filter.py --checkpoint models/clip_filter/best.pth --threshold 0.90
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
import torch
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

MANIFEST = Path(__file__).parent.parent / "data" / "mp_dataset_manifest.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RouteFinder-filter/1.0)"}


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


# ── DB ────────────────────────────────────────────────────────────────────────

def fetch_already_scored(conn, model_version):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT route_image_id FROM route_image_filter_scores WHERE filter_model_version = %s",
            (model_version,),
        )
        return {row[0] for row in cur.fetchall()}


def upsert_scores(conn, rows, model_version, threshold):
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO route_image_filter_scores
                    (route_image_id, filter_model_version, raw_score, passed, threshold)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (route_image_id, filter_model_version) DO UPDATE
                    SET raw_score = EXCLUDED.raw_score,
                        passed    = EXCLUDED.passed,
                        threshold = EXCLUDED.threshold,
                        scored_at = now()
                """,
                (row["route_image_id"], model_version, float(row["score"]),
                 bool(row["passed"]), threshold),
            )
    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--manifest",      default=str(MANIFEST))
    parser.add_argument("--model_version", default=None,
                        help="Tag stored in DB. Defaults to checkpoint filename stem.")
    parser.add_argument("--threshold",     type=float, default=0.8)
    parser.add_argument("--database_url",  default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--concurrency",   type=int, default=30)
    args = parser.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL not set — add it to .env or pass --database_url")

    model_version = args.model_version or Path(args.checkpoint).stem
    print(f"Filter model version: {model_version}  threshold: {args.threshold}")

    df = pd.read_csv(args.manifest)
    print(f"Manifest: {len(df)} images across {df['route_id'].nunique()} routes")

    conn = psycopg2.connect(args.database_url)
    already_scored = fetch_already_scored(conn, model_version)
    print(f"Already scored: {len(already_scored)} — skipping")

    todo = df[~df["route_image_id"].isin(already_scored)].reset_index(drop=True)
    print(f"To score: {len(todo)}")
    if todo.empty:
        print("Nothing to do.")
        conn.close()
        return

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    from clip_filter.model import load_model
    from clip_filter.dataset import VAL_TRANSFORM

    model, head, ckpt = load_model(args.checkpoint, device)
    model.eval()
    head.eval()
    print(f"Loaded checkpoint (epoch={ckpt['epoch']}, phase={ckpt['phase']})")

    scored_total = 0
    passed_total = 0
    skipped      = 0
    pending      = []
    chunk_size   = args.batch_size * 4

    for chunk_start in tqdm(range(0, len(todo), chunk_size), desc="Scoring", unit="chunk"):
        chunk    = todo.iloc[chunk_start: chunk_start + chunk_size]
        raw_data = asyncio.run(download_batch(chunk["image_url"].tolist(), args.concurrency))

        imgs, row_ids = [], []
        for (_, row), data in zip(chunk.iterrows(), raw_data):
            if data is None:
                skipped += 1
                continue
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
                imgs.append(VAL_TRANSFORM(img))
                row_ids.append(int(row["route_image_id"]))
            except Exception:
                skipped += 1

        if not imgs:
            continue

        with torch.no_grad():
            feats = model.encode_image(torch.stack(imgs).to(device)).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            scores = torch.sigmoid(head(feats).squeeze(1)).cpu().tolist()

        for rid, score in zip(row_ids, scores):
            pending.append({"route_image_id": rid, "score": score, "passed": score >= args.threshold})
            scored_total += 1
            passed_total += score >= args.threshold

        if len(pending) >= 500:
            upsert_scores(conn, pending, model_version, args.threshold)
            pending.clear()

    if pending:
        upsert_scores(conn, pending, model_version, args.threshold)

    conn.close()
    pct = 100 * passed_total / max(scored_total, 1)
    print(f"\nDone. Scored {scored_total}, skipped {skipped} failed downloads.")
    print(f"Passed threshold={args.threshold}: {passed_total} ({pct:.1f}%)")
    print(f"\nNext: python scripts/build_hf_dataset.py "
          f"--model_version {model_version} --threshold {args.threshold} --hf_repo <your/repo>")


if __name__ == "__main__":
    main()
