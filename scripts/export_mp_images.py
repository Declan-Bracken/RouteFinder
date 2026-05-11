"""
Export route_images from Postgres to a CSV for CLIP filtering on Kaggle.

Run this locally, commit the output CSV, then use it on Kaggle.

Usage:
    python scripts/export_mp_images.py
    git add data/mp_images_raw.csv && git commit -m "Export MP images for CLIP filtering"
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

OUT_PATH = Path(__file__).parent.parent / "data" / "mp_images_raw.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument(
        "--min_images_per_route",
        type=int,
        default=1,
        help="Only include routes with at least this many images (default: 1)",
    )
    args = parser.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL not set — add it to .env or pass --database_url")

    conn = psycopg2.connect(args.database_url)
    try:
        df = pd.read_sql(
            """
            SELECT
                ri.id,
                ri.route_id,
                ri.image_url,
                r.name  AS route_name,
                r.grade,
                a.name  AS area_name
            FROM route_images ri
            JOIN routes r ON r.id = ri.route_id
            JOIN areas  a ON a.id = r.area_id
            WHERE ri.image_url IS NOT NULL
              AND ri.image_url != ''
            ORDER BY ri.route_id, ri.id
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        sys.exit("No route_images found in DB.")

    # Optionally drop routes with too few images (not useful for contrastive training)
    if args.min_images_per_route > 1:
        counts = df.groupby("route_id")["id"].transform("count")
        before = len(df)
        df = df[counts >= args.min_images_per_route]
        print(f"Dropped {before - len(df)} rows (routes with < {args.min_images_per_route} images)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(
        f"Saved {len(df)} images across {df['route_id'].nunique()} routes "
        f"to {args.out}"
    )
    print("Next: git add data/mp_images_raw.csv && git commit && git push")
    print("Then run scripts/kaggle_clip_filter.py on Kaggle.")


if __name__ == "__main__":
    main()
