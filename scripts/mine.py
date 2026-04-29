"""
Filter multiview route images by embedding similarity to produce a cleaner training set.
Usage: python scripts/mine.py
"""
import asyncio
import gc
import logging
import pickle
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from routefinder.data.extract import load_flattened_tree
from routefinder.data.download import prefetch_cached_images
from routefinder.models.architectures import load_model, encode_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("img_downloader.log")],
)


def get_multiview_df(flattened_routes):
    """Return a DataFrame of routes that have >1 image, with integer route labels."""
    rows = []
    for route in flattened_routes:
        if len(route["images"]) > 1:
            str_label = f'{route["route_lineage"]}|{route["route_name"]}'
            for url in route["images"]:
                rows.append({
                    "url": url,
                    "route_name": route["route_name"],
                    "route_lineage": route["route_lineage"],
                    "str_label": str_label,
                })
    df = pd.DataFrame(rows)
    unique = sorted(df["str_label"].unique())
    df["label"] = df["str_label"].map({lbl: idx for idx, lbl in enumerate(unique)})
    return df


def is_dupe(img, seen_hashes):
    import hashlib
    h = hashlib.md5(img.tobytes()).hexdigest()
    if h in seen_hashes:
        return True, seen_hashes
    seen_hashes.add(h)
    return False, seen_hashes


def process_batch(downloaded, metadata_df, seen_hashes):
    url_to_row = {row["url"]: row for _, row in metadata_df.iterrows()}
    images, kept_rows = [], []
    for img, url in downloaded:
        dupe, seen_hashes = is_dupe(img, seen_hashes)
        if not dupe:
            images.append(img)
            kept_rows.append(url_to_row[url])
    return images, pd.DataFrame(kept_rows).reset_index(drop=True), seen_hashes


def mine_df(embeddings, df, similarity_threshold=0.5):
    """Keep only images that form at least one similar pair within their route group."""
    keep_indices = set()
    for _, group in df.groupby("label"):
        if len(group) < 2:
            continue
        indices = group.index.tolist()
        norm = torch.nn.functional.normalize(embeddings[indices], dim=1)
        sim = norm @ norm.T
        n = len(indices)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] > similarity_threshold:
                    keep_indices.add(indices[i])
                    keep_indices.add(indices[j])
    return df.loc[sorted(keep_indices)].copy()


def load_progress(path):
    master_df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    processed_file = path.with_suffix(".processed.txt")
    processed_batches = set()
    if processed_file.exists():
        with open(processed_file) as f:
            processed_batches = {int(x.strip()) for x in f}
    hashes_file = path.with_suffix(".hashes.pkl")
    seen_hashes = set()
    if hashes_file.exists():
        with open(hashes_file, "rb") as f:
            seen_hashes = pickle.load(f)
    return master_df, processed_batches, seen_hashes, hashes_file, processed_file


async def main(tree_path, model_path, batch_size, cache_dir, output_path, similarity_threshold=0.5, device="cpu"):
    cache_dir.mkdir(parents=True, exist_ok=True)
    master_df, processed_batches, seen_hashes, hashes_file, processed_file = load_progress(output_path)

    flattened = load_flattened_tree(tree_path)
    multiview_df = get_multiview_df(flattened)
    candidate_groups = [(label, group) for label, group in multiview_df.groupby("label")]
    n_routes_per_batch = max(1, int(batch_size / 2.5))
    semaphore = asyncio.Semaphore(50)

    model = load_model(model_path)
    model = model.to(device)

    for batch_idx, start in tqdm(
        enumerate(range(0, len(candidate_groups), n_routes_per_batch), start=1),
        desc="Mining batches",
    ):
        if batch_idx in processed_batches:
            continue

        batch_slice = candidate_groups[start:start + n_routes_per_batch]
        batch_meta = pd.concat([g for _, g in batch_slice], ignore_index=True)

        downloaded = await prefetch_cached_images(batch_meta["url"].tolist(), cache_dir, semaphore)
        logging.info(f"Batch {batch_idx}: downloaded {len(downloaded)} images")

        images, batch_meta, seen_hashes = process_batch(downloaded, batch_meta, seen_hashes)
        if not images:
            processed_batches.add(batch_idx)
            continue

        embeddings = encode_batch(images, model, device)
        keep_df = mine_df(embeddings, batch_meta, similarity_threshold)

        if len(keep_df) > 0:
            master_df = pd.concat([master_df, keep_df], ignore_index=True)
            master_df.to_csv(output_path, index=False)

        with open(processed_file, "a") as f:
            f.write(str(batch_idx) + "\n")
        with open(hashes_file, "wb") as f:
            pickle.dump(seen_hashes, f)

        logging.info(f"Batch {batch_idx}: kept {len(keep_df)} / {len(images)} images")
        del downloaded, images, embeddings
        if batch_idx % 10 == 0:
            gc.collect()
        for f in cache_dir.iterdir():
            f.unlink()

    logging.info("Mining complete")


if __name__ == "__main__":
    asyncio.run(main(
        tree_path="data/trees/mountain_project_tree.json.gz",
        model_path="models/pretrained/simclr.ckpt",
        batch_size=64,
        cache_dir=Path("data/cache"),
        output_path=Path("data/tagged_trees/mined_routes.csv"),
        similarity_threshold=0.6,
    ))
