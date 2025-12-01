from pathlib import Path
import asyncio
# from datasets import Dataset, Image as HFImage
import pandas as pd
import logging
from PIL import Image
import hashlib
import pickle
from tqdm import tqdm
import torch
import gc
import shutil
from src.data_curation.hf_dataset_creator import load_flattened_tree, prefetch_cached_images_limited
from src.utils.load_model import load_model, encode_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),               # Console output
        logging.FileHandler("img_downloader.log")     # Also save to file
    ]
)
    
# Get the list of routes with candidate positive groups.
def get_multiview(flattened_routes):
    multiview_images = []
    for route in flattened_routes:
        if len(route["images"]) > 1:
            for url in route["images"]:
                multiview_images.append({
                    "url" : url,
                    "route_name": route["route_name"],
                    "route_lineage": route["route_lineage"],
                    "str_label": f'{route["route_lineage"]}|{route["route_name"]}'
                })
    df = pd.DataFrame(multiview_images)
    # Map str → int
    unique = sorted(df["str_label"].unique())  # sorted so indices are deterministic
    label2id = {lbl: idx for idx, lbl in enumerate(unique)}
    df["label"] = df["str_label"].map(label2id)
    return df

def save_img(img, url, cache_dir):
    img = img.convert("RGB")
    filename = cache_dir / (url.split("/")[-1].split("?")[0] + ".jpg")
    img.save(filename, format="JPEG", quality=90, optimize = True, progressive = True)
    return filename

def is_dupe(img, seen_hashes = None):
    if seen_hashes is None:
        seen_hashes = set()
    # For hashing (fast): hash raw pixel bytes, not PNG encoding
    h = hashlib.md5(img.tobytes()).hexdigest()
    if h in seen_hashes:
        return True, seen_hashes
    seen_hashes.add(h)
    return False, seen_hashes

def process_batch(downloaded, metadata_df, seen_hashes=None):
    if seen_hashes is None:
        seen_hashes = set()

    images = []
    kept_metadata_rows = []

    # Use a dict for O(1) lookup from URL → metadata row
    url_to_row = {row["url"]: row for _, row in metadata_df.iterrows()}

    for img, url in downloaded:
        dupe, seen_hashes = is_dupe(img, seen_hashes)
        if dupe:
            continue
        
        images.append(img)
        kept_metadata_rows.append(url_to_row[url])

    # Build dataframe in same order as images
    filtered_metadata = pd.DataFrame(kept_metadata_rows)

    return images, filtered_metadata.reset_index(drop=True), seen_hashes

def mine_df(embeddings, df, similarity_threshold=0.5):
    keep_indices = set()

    for _, group in df.groupby('label'):
        if len(group) < 2:
            continue

        indices = group.index.to_list()
        group_embeddings = embeddings[indices]

        # Cosine similarity
        norm_embeds = torch.nn.functional.normalize(group_embeddings, dim=1)
        sim_matrix = norm_embeds @ norm_embeds.T

        n = len(indices)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > similarity_threshold:
                    keep_indices.add(indices[i])
                    keep_indices.add(indices[j])

    # Return filtered DataFrame
    return df.loc[sorted(keep_indices)].copy()

def load_progress(path):
    # Load previous curated data if exists
    if path.exists():
        master_df = pd.read_csv(path)
    else:
        master_df = pd.DataFrame()

    # Load processed batch list if exists
    processed_file = path.with_suffix(".processed.txt")
    if processed_file.exists():
        with open(processed_file, "r") as f:
            processed_batches = set(int(x.strip()) for x in f.readlines())
    else:
        processed_batches = set()
    
    # Load seen_hashes if exists
    hashes_file = path.with_suffix(".hashes.pkl")
    if hashes_file.exists():
        with open(hashes_file, "rb") as f:
            seen_hashes = pickle.load(f)
    else:
        seen_hashes = set()

    return master_df, processed_batches, seen_hashes, hashes_file, processed_file

async def main(tree_path, model_path, batch_size, cache_dir, curated_dataset_path, similarity_threshold = 0.5, device = "cpu"):
    # Create cache directory and load any progress.
    cache_dir.mkdir(parents=True, exist_ok=True)
    master_df, processed_batches, seen_hashes, hashes_file, processed_file = load_progress(curated_dataset_path)

    # Build label groups
    flattened_routes = load_flattened_tree(tree_path)
    multiview_routes = get_multiview(flattened_routes) # returns pd.DataFrame
    candidate_groups = [(label, group) for label, group in multiview_routes.groupby("label")]
    group_list = list(candidate_groups)
    n_routes_per_batch = max(1, int(batch_size / 2.5))
    
    # 4. Load model
    model = load_model(model_path)

    # 5. Start batching loop
    keep_samples = []
    downloaded_count = 0
    keep_count = 0

    for batch_idx, start in tqdm(enumerate(range(0, len(candidate_groups), n_routes_per_batch), start=1), desc="Processing Batches"):
        logging.info(f"Batch {batch_idx} in progress")

        # SKIP if already done
        if batch_idx in processed_batches:
            logging.info(f"Skipping batch {batch_idx} — already processed")
            continue
        
        batch_slice = group_list[start : start + n_routes_per_batch]
        batch_metadata = pd.concat([df for _, df in batch_slice], ignore_index=True)

        # Flatten URLs for all images in this batch
        urls = batch_metadata['url'].to_list()
        downloaded = await prefetch_cached_images_limited(urls, cache_dir) # Out: list of (PIL Image object, str)
        logging.info(f"Downloaded {len(downloaded)} images in batch {batch_idx}")

        # Hash and Drop Duplicates
        images, batch_metadata, seen_hashes = process_batch(downloaded, batch_metadata, seen_hashes) # out: list of PIL Images, metadata in pd dataframe, set
        if len(images) == 0:
            logging.info(f"Batch {batch_idx} produced 0 usable images — skip")
            processed_batches.add(batch_idx)
            continue

        downloaded_count += len(images)
        image_embeddings = encode_batch(images, model, device) # Out: 2d torch.tensor
        keep_df = mine_df(image_embeddings, batch_metadata, similarity_threshold) # Out: metadata df with only image samples with positive groups
        
        # 3.4 Add to dataset
        keep_samples.append(keep_df)
        keep_count += len(keep_df)

         # Append & save
        if len(keep_df) > 0:
            master_df = pd.concat([master_df, keep_df], ignore_index=True)
            master_df.to_csv(curated_dataset_path, index=False)

        # Register as processed
        with open(processed_file, "a") as f:
            f.write(str(batch_idx) + "\n")

        # Save seen_hashes checkpoint
        with open(hashes_file, "wb") as f:
            pickle.dump(seen_hashes, f)
        
        # 3.5 Clear cache of batch
        logging.info(f"✔ Batch {batch_idx} complete — added {len(keep_df)} curated samples out of {len(images)} imaged deduped")
        # end of batch
        del downloaded, images, image_embeddings   # free RAM
        if batch_idx % 10 == 0:
            gc.collect()  # only occasionally to avoid slowdown

        # clear cached JPGs on disk
        for f in cache_dir.iterdir():
            f.unlink()
        logging.info(f"Cache folder cleaned up for batch {start}.")
        break
    # 4. Save dataset (checkpoint)
    logging.info("🚀 Mining complete")

# Okay let's start by loading 1 image and 
if __name__ == "__main__":
    model_path = "models/pretrained/simclr.ckpt"
    tree_path = "src/data/trees/mountain_project_tree.json.gz"
    tree_name = tree_path.split("/")[-1].split(".")[0]
    cache_dir = Path("src/data/cache")
    curated_dataset_path = Path(f"src/data/curated_routes/{tree_name}")
    batch_size = 64 # very rough estimate, subject to change.
    sim_thresh = 0.6 # high precision threshold

    # try:
    asyncio.run(main(tree_path, 
                        model_path, 
                        batch_size, 
                        cache_dir,
                        curated_dataset_path, 
                        similarity_threshold = sim_thresh
                        ))
    
    # except:
    #     if cache_dir.exists():
    #         shutil.rmtree(cache_dir)
    #     logging.info("Cache folder cleaned up.")
