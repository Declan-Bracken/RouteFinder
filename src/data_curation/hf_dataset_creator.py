import asyncio, aiohttp
from pathlib import Path
from datasets import Dataset, Image as HFImage
import io
import gzip, json
import logging
from PIL import Image
import hashlib
from src.utils.extract import extract_with_lineage
import shutil
from tqdm import tqdm
import sys

# ---------------- Semaphore ----------------
CONCURRENCY_LIMIT = 50
BATCH_SIZE = 500
HF_REPO = "DeclanBracken/RouteFinderDataset"
PROCESSED_FILE = "src/data/uploaded_image_indices/processed_urls.json"
CACHE_DIR = Path("src/data/cache")

semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
COUNTER_LOCK = asyncio.Lock()

FAILED_COUNT = 0

# logging:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),               # Console output
        logging.FileHandler("img_downloader.log")     # Also save to file
    ]
)

# ---------------- Download limited number of images ----------------
def load_flattened_tree(file_path):
    with gzip.open(file_path, mode="rt", encoding="utf-8") as f:
        data = json.load(f)
        return extract_with_lineage(data)

# ---------------- Async download with size check ----------------
async def fetch_image_limited(session, url, cache_dir, semaphore, min_size=(224, 224)):
    global FAILED_COUNT
    async with semaphore:
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                content = await resp.read()
                img = Image.open(io.BytesIO(content))
                # Keep rgba images
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                # Skip small images
                if img.width < min_size[0] or img.height < min_size[1]:
                    return None, url

                # Save to cache
                filename = cache_dir / (url.split("/")[-1].split("?")[0])
                img.save(filename, format="JPEG", quality=90, optimize = True)
                return img, url
        except Exception as e:
            async with COUNTER_LOCK:
                FAILED_COUNT += 1
            logging.warning(f"Failed to fetch {url}: {e}") # honestly this is just messy
            # fallback or failure
            return None, url

async def prefetch_cached_images_limited(urls, cache_dir, semaphore, min_size=(224, 224)):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image_limited(session, url, cache_dir, semaphore, min_size) for url in urls]
        results = await asyncio.gather(*tasks)
        # Filter out failed / small downloads
        return [r for r in results if r[0] is not None]

# ---------------- Download N images with dedupe & size check ----------------
async def download_images(flattened_routes, cache_dir, semaphore, max_images=5000, batch_size = BATCH_SIZE):
    global FAILED_COUNT
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    total_added = 0

    # Flatten URLs up to max_images
    for route in flattened_routes:
        for img_url in route["images"]:
            if total_added >= max_images:
                break
            metadata.append({
                "route_name": route["route_name"],
                "route_lineage": route["route_lineage"],
                "url": img_url
            })
            total_added += 1
        if total_added >= max_images:
            break


    logging.info(f"Total URLs to download: {len(metadata)}")
    processed = load_processed()
    logging.info(f"Already processed: {len(processed)}")

    # track duplication across batches
    seen_hashes = set()

    # Track stats for this batch
    downloaded_count = 0
    dupe_count = 0

    # Process in batches
    for start in tqdm(range(0, len(metadata), batch_size), desc="Downloading batches"):
        batch_count = (start + 1)//BATCH_SIZE
        logging.info(f"Batch {start} in progress")
        batch = metadata[start:start+batch_size]
        batch = [x for x in batch if x["url"] not in processed] # Filter for samples which not been processed yet.

        if not batch:
            logging.warning("No new data to process")
            return

        urls = [x["url"] for x in batch]
        downloaded = await prefetch_cached_images_limited(urls, cache_dir, semaphore)
        logging.info(f"Completed downloading {len(downloaded)} images")

        batch_records = []
        for img, url in downloaded:
            meta = next(m for m in batch if m["url"] == url)

            img = img.convert("RGB")
            filename = cache_dir / (url.split("/")[-1].split("?")[0] + ".jpg")
            img.save(filename, format="JPEG", quality=90, optimize = True, progressive = True)

            # For hashing (fast): hash raw pixel bytes, not PNG encoding
            h = hashlib.md5(img.tobytes()).hexdigest()
            if h in seen_hashes:
                dupe_count += 1
                continue
            seen_hashes.add(h)
            downloaded_count += 1

            batch_records.append({
                **meta,
                "image": str(filename),     # File path, not bytes
            })

        logging.info(
            f"Batch # {batch_count}: "
            f"Downloaded {downloaded_count}, "
            f"Skipped duplicates {dupe_count}, "
            f"Failed URLs {FAILED_COUNT}"
        )

        if batch_records:
            save_processed(processed)
            repo_folder_name = f"batch_{start}"
            # Convert batch to HF Dataset and push
            batch_dataset = Dataset.from_list(batch_records)
            batch_dataset = batch_dataset.cast_column("image", HFImage())
            batch_dataset.push_to_hub(HF_REPO,
                                    data_dir = repo_folder_name,
                                    max_shard_size="1GB", 
                                    token=None)  # token optional if logged in
            logging.info(f"Batch pushed to new repo folder: {repo_folder_name}.")
            del batch_records, batch_dataset, downloaded  # free memory
            for f in cache_dir.iterdir():
                f.unlink()
            logging.info(f"Cache folder cleaned up for batch {start}.")
            processed.add(url)

    logging.info(f"All Images sucessfuly uploaded to HF!")

# For the processed file keeping track of progress.
def load_processed():
    if Path(PROCESSED_FILE).exists():
        return set(json.load(open(PROCESSED_FILE)))
    return set()

def save_processed(processed):
    json.dump(list(processed), open(PROCESSED_FILE, "w"))

# ---------------- Run ----------------
if __name__ == "__main__":
    tree_path = "src/data/trees/mountain_project_tree.json.gz"
    flattened_routes = load_flattened_tree(tree_path)
    try:
        asyncio.run(download_images(flattened_routes, CACHE_DIR, semaphore, max_images = 10000))
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        logging.info("Cache folder cleaned up.")
