"""
Export approved images from Postgres into a manifest CSV committed to the repo.

Images stay in B2. The manifest is just metadata:
    image_id, route_id, area_id, b2_key, label

Kaggle clones the repo and already has the manifest. It then downloads images
from B2 directly before training.

Usage:
    python scripts/export_dataset.py
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "manifest.csv"


def fetch_manifest(database_url: str) -> pd.DataFrame:
    conn = psycopg2.connect(database_url)
    try:
        df = pd.read_sql(
            """
            SELECT
                i.id        AS image_id,
                i.route_id,
                r.area_id,
                i.b2_key
            FROM images i
            JOIN routes r ON r.id = i.route_id
            WHERE i.status = 'approved'
            ORDER BY i.route_id, i.created_at
            """,
            conn,
        )
    finally:
        conn.close()

    df["label"] = df["route_id"]
    print(f"Manifest: {len(df)} images across {df['route_id'].nunique()} routes "
          f"in {df['area_id'].nunique()} areas")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--out", default=str(MANIFEST_PATH))
    args = parser.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL not set — add it to .env or pass --database_url")

    df = fetch_manifest(args.database_url)
    if df.empty:
        sys.exit("No approved images found.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved manifest to {args.out}")
    print("Commit and push data/manifest.csv to make it available on Kaggle.")


if __name__ == "__main__":
    main()
