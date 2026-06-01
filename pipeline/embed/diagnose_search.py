"""
Diagnose the embedding search pipeline using a known source image.

Given a local query image (the file you'd upload to /search):
  1. Apply search.py preprocessing (thumbnail if needed + JPEG encode/decode)
  2. Also try raw (no preprocessing)
  3. Embed both with the local model
  4. Fetch the stored embedding for each matching route from the DB
  5. Report cosine similarity

Usage:
    python scripts/diagnose_search.py --checkpoint path/to/model.ckpt \
        --image data/test_images/IMG_0107.jpeg \
        --route_name "20 Feet to France"
"""

import argparse
import io
import os
import sys
from pathlib import Path

import psycopg2
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "train"))


def search_preprocess(raw_bytes: bytes) -> Image.Image:
    """Exact search.py preprocessing."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.LANCZOS)
    return img


def raw_preprocess(raw_bytes: bytes) -> Image.Image:
    """No preprocessing — just decode."""
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def embed(model, transform, device, img: Image.Image) -> torch.Tensor:
    tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).squeeze(0).cpu()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="Path to your source query image")
    parser.add_argument("--route_name", default=None, help="Route name to look up (optional, shows top matches)")
    parser.add_argument("--model_version", default=os.environ.get("MODEL_VERSION", "v1"))
    parser.add_argument("--database_url", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL not set")

    from inference import RouteFinderModel, EVAL_TRANSFORM

    device = "cpu"
    print(f"Loading checkpoint: {args.checkpoint}")
    model = RouteFinderModel.load_from_checkpoint(
        args.checkpoint, map_location=device
    ).eval()

    raw_bytes = Path(args.image).read_bytes()
    img_pre = search_preprocess(raw_bytes)
    img_raw = raw_preprocess(raw_bytes)

    print(f"\nQuery image: {args.image}")
    print(f"  Original size : {Image.open(io.BytesIO(raw_bytes)).size}")
    print(f"  After preprocess size: {img_pre.size}")

    emb_pre = embed(model, EVAL_TRANSFORM, device, img_pre)
    emb_raw = embed(model, EVAL_TRANSFORM, device, img_raw)
    print(f"  Preprocessed vs raw embedding similarity: {cosine_sim(emb_pre, emb_raw):.6f}")

    # Fetch stored embeddings from DB
    conn = psycopg2.connect(args.database_url)
    with conn.cursor() as cur:
        if args.route_name:
            cur.execute(
                """
                SELECT r.name, e.embedding::text
                FROM image_embeddings e
                JOIN images i ON i.id = e.image_id
                JOIN routes r ON r.id = i.route_id
                WHERE e.model_version = %s
                  AND i.status = 'approved'
                  AND r.name ILIKE %s
                ORDER BY r.name
                """,
                (args.model_version, f"%{args.route_name}%"),
            )
        else:
            cur.execute(
                """
                SELECT r.name, e.embedding::text
                FROM image_embeddings e
                JOIN images i ON i.id = e.image_id
                JOIN routes r ON r.id = i.route_id
                WHERE e.model_version = %s
                  AND i.status = 'approved'
                ORDER BY r.name
                """,
                (args.model_version,),
            )
        rows = cur.fetchall()
    conn.close()

    print(f"\n{'Route':<35} {'w/ JPEG preprocess':>20} {'raw (no preprocess)':>20}")
    print("-" * 78)

    results = []
    for route_name, emb_str in rows:
        stored = torch.tensor([float(x) for x in emb_str.strip("[]").split(",")])
        sim_pre = cosine_sim(stored, emb_pre)
        sim_raw = cosine_sim(stored, emb_raw)
        results.append((sim_pre, sim_raw, route_name))

    results.sort(reverse=True)
    for sim_pre, sim_raw, route_name in results:
        flag = " ← raw wins" if sim_raw > sim_pre + 0.001 else ""
        print(f"{route_name[:34]:<35} {sim_pre:>20.6f} {sim_raw:>20.6f}{flag}")

    # Also compare the stored B2 image directly to its own stored embedding
    # This tells us whether the stored embeddings are self-consistent
    print("\n--- B2 self-consistency check (should be ~1.0) ---")
    import boto3
    from botocore.config import Config as BotoConfig
    b2 = boto3.client(
        "s3",
        endpoint_url=os.environ["B2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["B2_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
        config=BotoConfig(signature_version="s3v4"),
    )
    bucket = os.environ["B2_BUCKET_NAME"]

    conn2 = psycopg2.connect(args.database_url)
    with conn2.cursor() as cur:
        query = "%" + (args.route_name or "") + "%"
        cur.execute(
            """
            SELECT r.name, i.b2_key, e.embedding::text
            FROM image_embeddings e
            JOIN images i ON i.id = e.image_id
            JOIN routes r ON r.id = i.route_id
            WHERE e.model_version = %s AND i.status = 'approved'
              AND r.name ILIKE %s
            """,
            (args.model_version, query),
        )
        b2_rows = cur.fetchall()
    conn2.close()

    print(f"{'Route':<35} {'B2 self-sim':>12}")
    print("-" * 50)
    for route_name, b2_key, emb_str in b2_rows:
        stored = torch.tensor([float(x) for x in emb_str.strip("[]").split(",")])
        buf = io.BytesIO()
        b2.download_fileobj(bucket, b2_key, buf)
        b2_img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        emb_b2 = embed(model, EVAL_TRANSFORM, device, b2_img)
        sim = cosine_sim(stored, emb_b2)
        print(f"{route_name[:34]:<35} {sim:>12.6f}")


if __name__ == "__main__":
    main()
