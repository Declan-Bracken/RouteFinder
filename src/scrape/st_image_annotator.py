import asyncio
import aiohttp
import streamlit as st
import json
import io
import os
from PIL import Image
from pathlib import Path

# ============== CONFIG ==============
DATA_FILE = "src/data/trees/mount_nemo_tree.json"         # Your scraped JSON
TAGGED_FILE = "src/data/trees/tagged_mount_nemo_tree.json"
TAGS = ["approach", "on_wall", "action", "other"]
PREFETCH_LIMIT = 3  # Number of images to load ahead
# ===================================

"""
approach: A clear, identifying image of the route taken from the ground.
on_wall: A clear, identifying image of the route taken mid-climb.
action: An image taken from a third party somewhere on or beside the wall, mostly-person centric rather than climb centric.
other: Useless
"""

# -------- Recursive Image Extractor --------
def extract_routes(data):
    """Recursively extract route images and metadata from hierarchical data."""
    routes = []
    if isinstance(data, dict):
        if "routes" in data:
            for r in data["routes"]:
                routes.append({
                    "route_name": r.get("route_name"),
                    "area_name": data.get("area_name"),
                    "images": r.get("images", [])
                })
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                routes.extend(extract_routes(v))
    elif isinstance(data, list):
        for item in data:
            routes.extend(extract_routes(item))
    return routes

# -------- Async Image Prefetching --------
async def fetch_image(session, url):
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            content = await resp.read()
            return Image.open(io.BytesIO(content))
    except Exception:
        return None


async def prefetch_images(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# --- Load data ---
routes = json.load(open(DATA_FILE))
tagged = json.load(open(TAGGED_FILE)) if Path(TAGGED_FILE).exists() else {}

# -------- Main Streamlit App --------
st.title("🧗 Route Image Tagger (Async)")

# Load data
data = json.load(open(DATA_PATH))
routes = extract_routes(data)
all_images = [
    {"route": r["route_name"], "area": r["area_name"], "url": img}
    for r in routes for img in r["images"]
]

# Load existing tags
tagged = json.load(open(TAGGED_PATH)) if TAGGED_PATH.exists() else {}
untagged = [img for img in all_images if img["url"] not in tagged]

st.write(f"**Progress:** {len(tagged)} tagged / {len(all_images)} total")

if not untagged:
    st.success("🎉 All images tagged!")
    st.stop()

# Get current image
idx = st.session_state.get("index", 0)
if idx >= len(untagged):
    idx = 0

current = untagged[idx]

# Prefetch next few images asynchronously
next_urls = [img["url"] for img in untagged[idx:idx + PREFETCH_LIMIT]]
prefetched_images = asyncio.run(prefetch_images(next_urls))

# Display current image
current_img = prefetched_images[0]
if current_img:
    st.image(current_img, caption=f"{current['route']} ({current['area']})", use_container_width=True)
else:
    st.warning("⚠️ Could not load image.")
    if st.button("Skip"):
        st.session_state["index"] = idx + 1
        st.rerun()
    st.stop()

# Tag selection
selected_tags = st.multiselect("Select tags:", TAGS)

# Action buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Save Tags"):
        if not selected_tags:
            st.warning("Select at least one tag before saving.")
        else:
            tagged[current["url"]] = {
                "tags": selected_tags,
                "route": current["route"],
                "area": current["area"]
            }
            with open(TAGGED_PATH, "w") as f:
                json.dump(tagged, f, indent=2)
            st.success(f"Saved tags for {current['route']}")
            st.session_state["index"] = idx + 1
            st.rerun()

with col2:
    if st.button("⏭️ Skip"):
        st.session_state["index"] = idx + 1
        st.rerun()
