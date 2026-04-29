"""
Streamlit app for manually reviewing and accepting/rejecting route image pairs.
Usage: streamlit run scripts/annotate.py
"""
import asyncio
from pathlib import Path

import pandas as pd
import streamlit as st

from routefinder.data.download import prefetch_cached_images, load_image, remove_image_from_cache
from routefinder.data.extract import load_flattened_tree
from scripts.mine import get_multiview_df

DATA_PATH = Path("data/trees/mountain_project_tree.json.gz")
PROCESSED_PATH = Path("data/tagged_trees/processed_mountain_project_tree.csv")
CACHE_DIR = Path("data/cache")
PREFETCH_ROUTE_LIMIT = 10

CACHE_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data
def load_data(data_path):
    return get_multiview_df(load_flattened_tree(data_path))


@st.cache_data
def load_unprocessed_data(processed_path, all_images_df):
    if processed_path.exists():
        st.session_state.processed = pd.read_csv(processed_path)
        st.session_state.unprocessed = all_images_df[
            ~all_images_df["url"].isin(st.session_state.processed["url"])
        ]
    else:
        st.session_state.processed = pd.DataFrame()
        st.session_state.unprocessed = all_images_df

    all_labels = all_images_df.groupby("label")
    processed_labels = set(st.session_state.processed["label"].unique()) if len(st.session_state.processed) > 0 else set()
    unprocessed_labels = [lbl for lbl in all_labels.groups if lbl not in processed_labels]

    st.session_state.processed_labels = processed_labels
    st.session_state.unprocessed_labels = unprocessed_labels
    st.session_state.unprocessed_grouped = [g for _, g in st.session_state.unprocessed.groupby("label")]
    st.session_state.global_route_idx = 0
    st.session_state.local_route_idx = 0
    st.session_state.current_routes = []


st.title("Route Image Tagger")
all_images_df = load_data(DATA_PATH)
load_unprocessed_data(PROCESSED_PATH, all_images_df)

if len(st.session_state.unprocessed_labels) == st.session_state.global_route_idx:
    st.success("All images tagged!")
    for image_path in CACHE_DIR.glob("*.jpg"):
        remove_image_from_cache(str(image_path))
    st.stop()

n_processed = len(st.session_state.processed_labels)
n_total = len(st.session_state.unprocessed_labels) + n_processed
st.write(f"**Progress:** {n_processed + st.session_state.global_route_idx} tagged / {n_total} total")

n_cached = len(list(CACHE_DIR.glob("*.jpg")))
if n_cached == 0:
    st.session_state.current_routes = st.session_state.unprocessed_grouped[
        st.session_state.global_route_idx:st.session_state.global_route_idx + PREFETCH_ROUTE_LIMIT
    ]
    next_urls = [url for route in st.session_state.current_routes for url in route["url"]]
    st.session_state.local_route_idx = 0
    asyncio.run(prefetch_cached_images(next_urls, CACHE_DIR, asyncio.Semaphore(50)))

current_route_data = st.session_state.current_routes[st.session_state.local_route_idx]
current_urls = current_route_data["url"].tolist()
images = [load_image(url, CACHE_DIR) for url in current_urls]
cache_locations = [CACHE_DIR / (url.split("/")[-1].split("?")[0]) for url in current_urls]

st.subheader(f"Route {st.session_state.local_route_idx + 1} / {len(st.session_state.current_routes)}")
selected = set()
cols = st.columns(len(images))
for col, img, url, loc in zip(cols, images, current_urls, cache_locations):
    with col:
        if img is None:
            cache_locations.remove(loc)
            continue
        st.image(img)
        if st.checkbox("Keep", key=url):
            selected.add(url)

if st.button("Next Route"):
    new_rows = current_route_data.copy()
    new_rows["keep"] = current_route_data["url"].isin(selected)
    new_rows.to_csv(PROCESSED_PATH, mode="a", header=not PROCESSED_PATH.exists(), index=False)
    for loc in cache_locations:
        remove_image_from_cache(loc)

    st.session_state.global_route_idx += 1
    st.session_state.local_route_idx += 1
    if st.session_state.global_route_idx >= len(st.session_state.unprocessed_labels):
        st.success("All routes processed!")
    else:
        st.rerun()
