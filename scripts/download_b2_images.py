"""
Download B2 images for a given route to compare visually.

Usage:
    python scripts/download_b2_images.py --route_name "20 Feet to France"
"""

import argparse
import io
import os
import sys
from pathlib import Path

import boto3
import psycopg2
from botocore.config import Config as BotoConfig
from PIL import Image, ImageOps
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route_name", default=None, help="Filter by route name (omit to download all)")
    parser.add_argument("--model_version", default=os.environ.get("MODEL_VERSION", "v1"))
    parser.add_argument("--database_url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--out_dir", default="data/test_images/b2_downloads")
    args = parser.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL not set")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(args.database_url)
    with conn.cursor() as cur:
        if args.route_name:
            cur.execute(
                """
                SELECT i.id, i.b2_key, r.name
                FROM images i
                JOIN routes r ON r.id = i.route_id
                WHERE i.status = 'approved' AND r.name ILIKE %s
                """,
                (f"%{args.route_name}%",),
            )
        else:
            cur.execute(
                """
                SELECT i.id, i.b2_key, r.name
                FROM images i
                JOIN routes r ON r.id = i.route_id
                WHERE i.status = 'approved'
                ORDER BY r.name
                """
            )
        rows = cur.fetchall()
    conn.close()

    b2 = boto3.client(
        "s3",
        endpoint_url=os.environ["B2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["B2_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
        config=BotoConfig(signature_version="s3v4"),
    )
    bucket = os.environ["B2_BUCKET_NAME"]

    print(f"Downloading {len(rows)} image(s) for '{args.route_name}'...")
    for image_id, b2_key, route_name in rows:
        buf = io.BytesIO()
        b2.download_fileobj(bucket, b2_key, buf)
        dest = out / f"{image_id}.jpg"
        dest.write_bytes(buf.getvalue())

        img = Image.open(io.BytesIO(buf.getvalue()))
        print(f"  {dest}  size={img.size}")

    # Also show info about the local query image
    query = Path("data/test_images/IMG_0107.jpeg")
    if query.exists():
        img = Image.open(query)
        exif = img.getexif()
        orientation = exif.get(274, 1)  # 274 = Orientation tag
        print(f"\nQuery image: {query}  size={img.size}  EXIF orientation={orientation}")
        # Save EXIF-corrected version for comparison
        corrected = ImageOps.exif_transpose(img)
        corrected_path = out / "query_exif_corrected.jpg"
        corrected.save(corrected_path, format="JPEG", quality=95)
        print(f"EXIF-corrected query saved to: {corrected_path}  size={corrected.size}")

    print(f"\nOpen {out}/ to visually compare the B2 images with your query image.")


if __name__ == "__main__":
    main()
