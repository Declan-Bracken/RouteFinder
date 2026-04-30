"""
Export approved images from Postgres + B2 into an HF dataset.

Images stay in B2. What gets pushed to HF is a manifest-only dataset:
    image_id, route_id, area_id, b2_key, label

At training time Kaggle reads the manifest and downloads images from B2 directly.

Usage:
    python scripts/export_dataset.py \
        --hf_dataset DeclanBracken/RouteFinderDatasetV3 \
        --hf_token $HF_TOKEN
"""

import argparse
import os
import sys

import pandas as pd
import psycopg2
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()


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

    # label = route_id directly — SupCon just needs consistent integers per class
    df["label"] = df["route_id"]
    print(f"Manifest: {len(df)} images across {df['route_id'].nunique()} routes "
          f"in {df['area_id'].nunique()} areas")
    return df


def push_to_hf(df: pd.DataFrame, hf_dataset: str, hf_token: str):
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(hf_dataset, token=hf_token)
    print(f"Pushed manifest to {hf_dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", required=True)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--database_url", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL not set — add it to .env or pass --database_url")
    if not args.hf_token:
        sys.exit("HF_TOKEN not set — add it to .env or pass --hf_token")

    df = fetch_manifest(args.database_url)
    if df.empty:
        sys.exit("No approved images found. Submit and approve some images first.")

    push_to_hf(df, args.hf_dataset, args.hf_token)


if __name__ == "__main__":
    main()
