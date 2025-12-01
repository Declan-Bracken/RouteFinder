import asyncio
import streamlit as st
from pathlib import Path
import pandas as pd

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# print(sys.path)
from src.utils.download_imgs import prefetch_cached_images, load_image, remove_image_from_cache
from src.data_curation.hf_dataset_creator import load_flattened_tree
from src.data_curation.multiview_data_mining import get_multiview
import shutil

# ============== CONFIG ==============
DATA_PATH = Path(f"src/data/trees/mountain_project_tree.json.gz")
PROCESSED_PATH = Path(f"src/data/tagged_trees/processed_mountain_project_tree.csv")
CACHE_DIR = Path("src/data/cache")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PREFETCH_ROUTE_LIMIT = 10  # Number of images to load ahead
# ===================================

@st.cache_data
def load_data(DATA_PATH):
    # Load data
    flattened_routes = load_flattened_tree(DATA_PATH)
    multiview_routes = get_multiview(flattened_routes)
    return multiview_routes

@st.cache_data
def load_unprocessed_data(PROCESSED_PATH, all_images_df):
    # Runs once at the beginning of script start
    if PROCESSED_PATH.exists():
        print("processed_path exists")
        st.session_state.processed = pd.read_csv(PROCESSED_PATH)
        st.session_state.unprocessed = all_images_df[~all_images_df["url"].isin(st.session_state.processed["url"])]
        st.session_state.global_route_idx = st.session_state.processed.iloc[-1]["label"] + 1
    else:
        st.session_state.processed = pd.DataFrame()
        st.session_state.unprocessed = all_images_df

    # Build unprocessed ONCE
    all_labels = all_images_df.groupby("label")
    if len(st.session_state.processed) > 0:
        st.session_state.processed_labels = set(st.session_state.processed["label"].unique())
    else:
        st.session_state.processed_labels = set()
    unprocessed_labels = [label for label in all_labels.groups.keys() if label not in st.session_state.processed_labels]

    st.session_state.unprocessed_labels = unprocessed_labels
    st.session_state.unprocessed_grouped = [g for _,g in st.session_state.unprocessed.groupby("label")]
    st.session_state.global_route_idx = 0
    st.session_state.local_route_idx = 0
    st.session_state.current_routes = []
    
# -------- Main Streamlit App --------
st.title("🧗 Route Image Tagger")
all_images_df = load_data(DATA_PATH)
load_unprocessed_data(PROCESSED_PATH, all_images_df)

if len(st.session_state.unprocessed_labels) == st.session_state.global_route_idx:
    st.success("🎉 All images tagged!")
    st.text("Clearing Image Cache...")
    for image_path in list(CACHE_DIR.glob("*.jpg")):
        remove_image_from_cache(str(image_path))
    st.stop()

st.write(f"**Progress:** {len(st.session_state.processed_labels) + st.session_state.global_route_idx} tagged / {len(st.session_state.unprocessed_labels) + len(st.session_state.processed_labels)} total")

# Retrieve the number of images currently cached
n_cached_images = len(list(CACHE_DIR.glob("*.jpg")))
if n_cached_images == 0: # No images in cache?
    # create routes
    st.session_state.current_routes = st.session_state.unprocessed_grouped[
        st.session_state.global_route_idx : st.session_state.global_route_idx + PREFETCH_ROUTE_LIMIT
    ]
    # grab all necessary image urls
    next_urls = [url for route in st.session_state.current_routes for url in route["url"]]
    st.session_state.local_route_idx = 0
    # Limit concurrency to 50 simultaneous downloads
    semaphore = asyncio.Semaphore(50)
    _ = asyncio.run(prefetch_cached_images(next_urls, CACHE_DIR, semaphore))

print(f"\n\nCache Size: {n_cached_images}\nLength of current_routes: {len(st.session_state.current_routes)}\nCurrent Route Idx: {st.session_state.local_route_idx}\n\n")

# Get the current batch of images for the current route and their cache locations
current_route_data = st.session_state.current_routes[st.session_state.local_route_idx]
current_urls = current_route_data["url"].to_list() # grab all sample images from current route
images = [load_image(current_urls[i], CACHE_DIR) for i in range(len(current_urls))]
image_cache_locations = [CACHE_DIR / (current_urls[i].split("/")[-1].split("?")[0]) for i in range(len(current_urls))]

# ---------- Display images in columns with checkboxes ----------
st.subheader(f"Current Batch, Route {st.session_state.local_route_idx+1} / {len(st.session_state.current_routes)}")
selected = set()
cols = st.columns(len(images))
for col, img, url in zip(cols, images, current_urls):
    with col:
        if img is None:
            image_cache_locations.remove(CACHE_DIR / (url.split("/")[-1].split("?")[0]))
            continue
        # if img.
        st.image(img)
        if st.checkbox("Keep", key=url):
            selected.add(url)

# ---------- Navigation ----------
if st.button("Next Route"):
    # Save selected samples to processed dataframe, include new column to specify which should be kept.
    new_rows = current_route_data
    new_rows["keep"] = current_route_data["url"].isin(selected)
    new_rows.to_csv(
        PROCESSED_PATH,
        mode="a",
        header=not PROCESSED_PATH.exists(),
        index=False,
    )
    print(image_cache_locations)
    _ = [remove_image_from_cache(location) for location in image_cache_locations]

    # Increment route index (global and local)
    st.session_state.global_route_idx += 1
    st.session_state.local_route_idx += 1
    if st.session_state.global_route_idx >= len(st.session_state.unprocessed_labels):
        st.success("🎉 All routes processed!")
    else:
        st.rerun()
