"""
Embed CLIP-filtered MP images and write to route_image_embeddings for bootstrap search coverage.

Downloads each URL directly (no B2), embeds with the RouteFinderModel checkpoint,
and upserts into route_image_embeddings. After this runs, /search returns results
for any route with MP images — no field submissions required.

Prerequisites:
    - Run app/migrations/003_route_image_embeddings.sql against the live DB
    - Have data/mp_images_filtered.csv (output of kaggle_clip_filter.py)
    - Have a model checkpoint (download from B2 or Kaggle output)

Usage:
    python scripts/bootstrap_embeddings.py --checkpoint models/model.ckpt
"""

import argparse
import asyncio
import io
import os
import sys
from pathlib import Path

import aiohttp
import psycopg2
import torch
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent / "train"))

FILTERED_CSV = Path(__file__).parent.parent / "data" / "mp_images_filtered.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RouteFinder-bootstrap/1.0)"}


# ── Download ──────────────────────────────────────────────────────────────────

async def _fetch(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> bytes | None:
    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    if resp.status in (403, 404, 410):
                        return None
            except Exception:
                pass
            await asyncio.sleep(2 ** attempt)
    return None


async def download_batch(urls: list[str], concurrency: int = 30) -> list[bytes | None]:
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        return await asyncio.gather(*[_fetch(session, url, sem) for url in urls])


# ── DB helpers ────────────────────────────────────────────────────────────────

def fetch_already_embedded(conn, model_version: str) -> set[int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT route_image_id FROM route_image_embeddings WHERE model_version = %s",
            (model_version,),
        )
        return {row[0] for row in cur.fetchall()}


def upsert_embeddings(conn, rows: list[dict], model_version: str):
    with conn.cursor() as cur:
        for row in rows:
            vec_str = "[" + ",".join(f"{x:.8f}" for x in row["embedding"]) + "]"
            cur.execute(
                """
                INSERT INTO route_image_embeddings (route_image_id, model_version, embedding)
                VALUES (%s, %s, %s::vector)
                ON CONFLICT (route_image_id, model_version) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        computed_at = now()
                """,
                (row["route_image_id"], model_version, vec_str),
            )
    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--filtered_csv", default=str(FILTERED_CSV))
    parser.add_argument("--model_version", default=os.environ.get("MODEL_VERSION", "v1"))
    parser.add_argument("--database_url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--concurrency",  type=int, default=30)
    args = parser.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL not set — add it to .env or pass --database_url")

    import pandas as pd
    df = pd.read_csv(args.filtered_csv)
    keep = df[df["keep"] == True].reset_index(drop=True)
    print(f"Filtered CSV: {len(df)} total, {len(keep)} marked keep=True")

    from inference import RouteFinderModel, EVAL_TRANSFORM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.checkpoint} on {device}")
    model = RouteFinderModel.load_from_checkpoint(
        args.checkpoint, map_location=device
    ).eval()

    conn = psycopg2.connect(args.database_url)
    done = fetch_already_embedded(conn, args.model_version)
    print(f"Already embedded: {len(done)}")

    todo = keep[~keep["id"].isin(done)].reset_index(drop=True)
    print(f"To embed: {len(todo)}")
    if todo.empty:
        print("Nothing to do.")
        conn.close()
        return

    batch: list[tuple[int, torch.Tensor]] = []
    skipped = 0

    for chunk_start in tqdm(range(0, len(todo), args.batch_size * 4), desc="Downloading"):
        chunk = todo.iloc[chunk_start : chunk_start + args.batch_size * 4]
        raw = asyncio.run(download_batch(chunk["image_url"].tolist(), args.concurrency))

        for (_, row), data in zip(chunk.iterrows(), raw):
            if data is None:
                skipped += 1
                continue
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                skipped += 1
                continue
            batch.append((int(row["id"]), EVAL_TRANSFORM(img).unsqueeze(0)))

            if len(batch) >= args.batch_size:
                _flush(batch, model, device, conn, args.model_version)
                batch = []

    if batch:
        _flush(batch, model, device, conn, args.model_version)

    conn.close()
    print(f"\nDone — embedded {len(todo) - skipped} images (skipped {skipped} failed downloads)")
    print(f"Model version: {args.model_version}")
    print("\nBootstrap complete. /search now returns results for MP routes.")
    print("Consider running the IVFFlat index creation (see migration 003) once >1000 rows.")


def _flush(batch, model, device, conn, model_version):
    ids, tensors = zip(*batch)
    with torch.no_grad():
        embs = model(torch.cat(tensors).to(device)).cpu().tolist()
    upsert_embeddings(
        conn,
        [{"route_image_id": id_, "embedding": emb} for id_, emb in zip(ids, embs)],
        model_version,
    )
    batch.clear()


if __name__ == "__main__":
    main()
