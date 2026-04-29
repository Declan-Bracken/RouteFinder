"""
Download route images from the MP tree and upload them to a HuggingFace dataset.
Usage: python scripts/curate.py
"""
import asyncio
import hashlib
import json
import logging
import shutil
from pathlib import Path

from datasets import Dataset, Image as HFImage
from tqdm import tqdm

from routefinder.data.extract import load_flattened_tree
from routefinder.data.download import prefetch_cached_images

HF_REPO = "DeclanBracken/RouteFinderDataset"
PROCESSED_FILE = "data/uploaded_image_indices/processed_urls.json"
CACHE_DIR = Path("data/cache")
CONCURRENCY_LIMIT = 50
BATCH_SIZE = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("img_downloader.log")],
)


def load_processed():
    if Path(PROCESSED_FILE).exists():
        return set(json.load(open(PROCESSED_FILE)))
    return set()


def save_processed(processed):
    Path(PROCESSED_FILE).parent.mkdir(parents=True, exist_ok=True)
    json.dump(list(processed), open(PROCESSED_FILE, "w"))


async def download_images(flattened_routes, cache_dir, max_images=5000, batch_size=BATCH_SIZE):
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for route in flattened_routes:
        for img_url in route["images"]:
            if len(metadata) >= max_images:
                break
            metadata.append({
                "route_name": route["route_name"],
                "route_lineage": route["route_lineage"],
                "url": img_url,
            })
        if len(metadata) >= max_images:
            break

    logging.info(f"Total URLs to download: {len(metadata)}")
    processed = load_processed()
    logging.info(f"Already processed: {len(processed)}")
    seen_hashes = set()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    for start in tqdm(range(0, len(metadata), batch_size), desc="Downloading batches"):
        batch = [x for x in metadata[start:start + batch_size] if x["url"] not in processed]
        if not batch:
            logging.warning("No new data in batch")
            continue

        downloaded = await prefetch_cached_images([x["url"] for x in batch], cache_dir, semaphore)
        logging.info(f"Batch {start}: downloaded {len(downloaded)} images")

        url_to_meta = {x["url"]: x for x in batch}
        batch_records = []
        dupe_count = 0

        for img, url in downloaded:
            img = img.convert("RGB")
            h = hashlib.md5(img.tobytes()).hexdigest()
            if h in seen_hashes:
                dupe_count += 1
                continue
            seen_hashes.add(h)

            filename = cache_dir / (url.split("/")[-1].split("?")[0] + ".jpg")
            img.save(filename, format="JPEG", quality=90, optimize=True, progressive=True)
            batch_records.append({**url_to_meta[url], "image": str(filename)})

        logging.info(f"  kept {len(batch_records)}, dupes {dupe_count}")

        if batch_records:
            ds = Dataset.from_list(batch_records).cast_column("image", HFImage())
            ds.push_to_hub(HF_REPO, data_dir=f"batch_{start}", max_shard_size="1GB", token=None)
            logging.info(f"  pushed batch_{start} to HF")
            for f in cache_dir.iterdir():
                f.unlink()

        for rec in batch_records:
            processed.add(rec["url"])
        save_processed(processed)

    logging.info("All images uploaded to HF")


if __name__ == "__main__":
    tree_path = "data/trees/mountain_project_tree.json.gz"
    flattened = load_flattened_tree(tree_path)
    try:
        asyncio.run(download_images(flattened, CACHE_DIR, max_images=10000))
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        logging.info("Cache cleaned up")
