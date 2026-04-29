"""
Upload a curated CSV of route images to a HuggingFace dataset.
Usage: python scripts/upload_curated.py
"""
import asyncio
import gc
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset, Image as HFImage
from tqdm import tqdm

from routefinder.data.download import prefetch_cached_images

HF_REPO = "DeclanBracken/RouteFinderDatasetV2"
CONCURRENCY_LIMIT = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("hf_dataset_creator.log")],
)


async def main(curated_csv, route_batch_size, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    dataset = pd.read_csv(curated_csv)
    keep = dataset[dataset["keep"] == True]
    groups = [g for _, g in keep.groupby("label") if len(g) > 1]

    downloaded_count = 0
    for batch_idx, start in tqdm(
        enumerate(range(0, len(groups), route_batch_size), start=1),
        desc="Uploading batches",
    ):
        batch_meta = pd.concat(groups[start:start + route_batch_size], ignore_index=True)
        urls = batch_meta["url"].tolist()
        downloaded = await prefetch_cached_images(urls, cache_dir, semaphore)

        batch_meta["image"] = str(cache_dir) + "/" + (
            batch_meta["url"].str.split("/").str[-1].str.split("?").str[0]
        )
        batch_meta = batch_meta[batch_meta["image"].apply(lambda p: Path(p).is_file())].reset_index(drop=True)

        if len(downloaded) == 0:
            logging.info(f"Batch {batch_idx}: no usable images, skipping")
            continue

        downloaded_count += len(downloaded)
        logging.info(f"Batch {batch_idx}: {downloaded_count} total downloaded")

        ds = Dataset.from_pandas(batch_meta)
        _ = ds[:]["image"]  # force full read before cache clear
        ds = ds.cast_column("image", HFImage())
        ds.push_to_hub(HF_REPO, data_dir=f"batch_{batch_idx}", max_shard_size="1GB", token=None)
        logging.info(f"Batch {batch_idx} pushed to HF")

        del downloaded
        if batch_idx % 2 == 0:
            gc.collect()
        for f in cache_dir.iterdir():
            f.unlink()

    logging.info("Dataset upload complete")


if __name__ == "__main__":
    asyncio.run(main(
        curated_csv=Path("data/tagged_trees/processed_mountain_project_tree.csv"),
        route_batch_size=50,
        cache_dir=Path("data/cache"),
    ))
