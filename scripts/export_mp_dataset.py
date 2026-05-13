"""
Export route_images from Postgres with full metadata for CLIP filtering and HF dataset creation.

Unlike export_mp_images.py (used for bootstrap embeddings), this script pulls all route
metadata (type, description, location, gps) and the full area hierarchy path via a
recursive CTE. Output CSV is the manifest consumed by apply_clip_filter.py.

Usage:
    python scripts/export_mp_dataset.py
    python scripts/export_mp_dataset.py --min_images_per_route 2 --out data/mp_dataset_manifest.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

OUT_PATH = Path(__file__).parent.parent / "data" / "mp_dataset_manifest.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--min_images_per_route", type=int, default=2,
                        help="Only include routes with at least N images (default: 2, for positive pairs)")
    args = parser.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL not set — add it to .env or pass --database_url")

    conn = psycopg2.connect(args.database_url)
    try:
        df = pd.read_sql(
            """
            WITH RECURSIVE area_tree AS (
                SELECT id, name, parent_id,
                       name::text AS area_path,
                       0          AS area_depth
                FROM areas
                WHERE parent_id IS NULL
                UNION ALL
                SELECT a.id, a.name, a.parent_id,
                       (t.area_path || '/' || a.name),
                       t.area_depth + 1
                FROM areas a
                JOIN area_tree t ON a.parent_id = t.id
            )
            SELECT
                ri.id           AS route_image_id,
                ri.image_url,
                r.id            AS route_id,
                r.name          AS route_name,
                r.grade,
                r.type,
                r.description,
                r.location,
                r.gps,
                a.id            AS area_id,
                a.name          AS area_name,
                t.area_path,
                t.area_depth
            FROM route_images ri
            JOIN routes    r ON r.id  = ri.route_id
            JOIN areas     a ON a.id  = r.area_id
            JOIN area_tree t ON t.id  = a.id
            WHERE ri.image_url IS NOT NULL
              AND ri.image_url != ''
            ORDER BY r.area_id, r.id, ri.id
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        sys.exit("No route_images found in DB.")

    before = len(df)
    counts = df.groupby("route_id")["route_image_id"].transform("count")
    df = df[counts >= args.min_images_per_route].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows (routes with < {args.min_images_per_route} images)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(
        f"Saved {len(df)} images across {df['route_id'].nunique()} routes "
        f"and {df['area_id'].nunique()} areas to {args.out}"
    )
    print(f"Area depth range: {df['area_depth'].min()} - {df['area_depth'].max()}")
    print(f"\nNext: python scripts/apply_clip_filter.py --checkpoint models/clip_filter/best.pth")


if __name__ == "__main__":
    main()
