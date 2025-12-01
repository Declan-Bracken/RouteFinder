import asyncio
from pathlib import Path
from datasets import Dataset, Image as HFImage
import logging
from tqdm import tqdm
import pandas as pd
import gc
import os
from src.data_curation.hf_dataset_creator import prefetch_cached_images_limited

HF_REPO = "DeclanBracken/RouteFinderDatasetV2"
COUNTER_LOCK = asyncio.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),               # Console output
        logging.FileHandler("hf_dataset_creator.log")     # Also save to file
    ]
)

async def main(curated_dataset_path, route_batch_size, cache_dir, semaphore):
    # Create cache directory and load any progress.
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build label groups
    dataset = pd.read_csv(curated_dataset_path)
    # print(len(dataset))
    dataset_keep = dataset[dataset["keep"] == True] # forgetting this one line wrecks me lol.
    # print(len(dataset_keep))
    # return
    ds_groups = [g for _, g in dataset_keep.groupby("label") if len(g) > 1] # take only routes with greate than 1 image sample

    # Start batching loop
    downloaded_count = 0
    for batch_idx, start in tqdm(enumerate(range(0, len(ds_groups), route_batch_size), start=1), desc="Processing Batches"):
        logging.info(f"Batch {batch_idx} in progress")
        
        batch_slice = ds_groups[start : start + route_batch_size]
        batch_metadata = pd.concat([df for df in batch_slice], ignore_index=True)

        # Flatten URLs for all images in this batch
        urls = batch_metadata['url'].to_list()
        print(f"Length of URLs list: {len(urls)}")
        downloaded = await prefetch_cached_images_limited(urls, cache_dir, semaphore) # Out: list of (PIL Image object, str)
        batch_metadata["image"] = str(cache_dir) + "/" + (batch_metadata['url'].str.split("/").str[-1].str.split("?").str[0])
        # remove images from dataset if not cached:
        batch_metadata = batch_metadata[batch_metadata["image"].apply(lambda p: Path(p).is_file())].reset_index(drop=True)

        if len(downloaded) == 0:
            logging.info(f"Batch {batch_idx} produced 0 usable images — skip")
            continue

        # n_cached = len([name for name in os.listdir(cache_dir)
        #         if os.path.isfile(os.path.join(cache_dir, name))])
        # assert len(batch_metadata) == n_cached, f"Number of listed samples does not match number of cached images: {len(batch_metadata)} : {n_cached}"

        downloaded_count += len(downloaded)

        logging.info(
            f"Batch # {batch_idx}: "
            f"Downloaded {downloaded_count}"
        )

        repo_folder_name = f"batch_{batch_idx}"
        # Convert batch to HF Dataset and push
        batch_dataset = Dataset.from_pandas(batch_metadata)
        # Force dataset to actually read all images BEFORE deleting cache
        _ = batch_dataset[:]["image"]     # <-- forces full resolution and prevents lazy loading

        batch_dataset = batch_dataset.cast_column("image", HFImage())
        batch_dataset.push_to_hub(HF_REPO,
                                data_dir = repo_folder_name,
                                max_shard_size="1GB", 
                                token=None)  # token optional if logged in
        logging.info(f"Batch pushed to new repo folder: {repo_folder_name}.")
        # end of batch
        del downloaded
        if batch_idx % 2 == 0:
            gc.collect()  # only occasionally to avoid slowdown

        # clear cached JPGs on disk
        for f in cache_dir.iterdir():
            f.unlink()
        logging.info(f"Cache folder cleaned up for batch {start}.")
    # 4. Save dataset (checkpoint)
    logging.info("🚀 Dataset upload complete")

# Okay let's start by loading 1 image and 
if __name__ == "__main__":

    cache_dir = Path("src/data/cache")
    curated_dataset_path = Path(f"src/data/tagged_trees/processed_mountain_project_tree.csv")
    route_batch_size = 50 # very rough estimate, subject to change.
    CONCURRENCY_LIMIT = 50
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    asyncio.run(main(curated_dataset_path,
                        route_batch_size, 
                        cache_dir,
                        semaphore
                        ))
