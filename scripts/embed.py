"""
Compute embeddings for all approved images and write to image_embeddings table.

Run this locally after downloading a checkpoint from Kaggle.

Usage:
    python scripts/embed.py --checkpoint path/to/model.ckpt --model_version v1
"""

import argparse
import io
import os
import sys
from pathlib import Path

import boto3
import psycopg2
import torch
from botocore.config import Config as BotoConfig
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "train"))


def _b2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["B2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["B2_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
        config=BotoConfig(signature_version="s3v4"),
    )


def fetch_pending(conn, model_version: str) -> list[dict]:
    """Approved images that don't yet have an embedding for this model version."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT i.id, i.b2_key
            FROM images i
            WHERE i.status = 'approved'
              AND NOT EXISTS (
                  SELECT 1 FROM image_embeddings e
                  WHERE e.image_id = i.id AND e.model_version = %s
              )
            """,
            (model_version,),
        )
        return [{"id": str(row[0]), "b2_key": row[1]} for row in cur.fetchall()]


def download_image(b2, bucket: str, key: str) -> Image.Image:
    buf = io.BytesIO()
    b2.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def insert_embeddings(conn, rows: list[dict], model_version: str):
    with conn.cursor() as cur:
        for row in rows:
            vec_str = "[" + ",".join(f"{x:.8f}" for x in row["embedding"]) + "]"
            cur.execute(
                """
                INSERT INTO image_embeddings (image_id, model_version, embedding)
                VALUES (%s, %s, %s::vector)
                ON CONFLICT (image_id, model_version) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        computed_at = now()
                """,
                (row["image_id"], model_version, vec_str),
            )
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_version", default=os.environ.get("MODEL_VERSION", "v1"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--database_url", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL not set — add it to .env or pass --database_url")

    from train import RouteFinderModel, EVAL_TRANSFORM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.checkpoint} on {device}")
    model = RouteFinderModel.load_from_checkpoint(
        args.checkpoint, map_location=device
    ).eval()

    conn = psycopg2.connect(args.database_url)
    b2 = _b2_client()
    bucket = os.environ["B2_BUCKET_NAME"]

    pending = fetch_pending(conn, args.model_version)
    print(f"Images to embed: {len(pending)}")
    if not pending:
        print("Nothing to do — all approved images already have embeddings for this version.")
        conn.close()
        return

    batch = []
    for item in tqdm(pending, desc="Embedding"):
        try:
            img = download_image(b2, bucket, item["b2_key"])
        except Exception as e:
            print(f"  Skipping {item['id']}: {e}")
            continue

        batch.append((item["id"], EVAL_TRANSFORM(img).unsqueeze(0)))

        if len(batch) >= args.batch_size:
            _flush(batch, model, device, conn, args.model_version)
            batch = []

    if batch:
        _flush(batch, model, device, conn, args.model_version)

    conn.close()
    print(f"Done — {len(pending)} images embedded as '{args.model_version}'")


def _flush(batch, model, device, conn, model_version):
    ids, tensors = zip(*batch)
    with torch.no_grad():
        embs = model(torch.cat(tensors).to(device)).cpu().tolist()
    insert_embeddings(
        conn,
        [{"image_id": id_, "embedding": emb} for id_, emb in zip(ids, embs)],
        model_version,
    )


if __name__ == "__main__":
    main()
