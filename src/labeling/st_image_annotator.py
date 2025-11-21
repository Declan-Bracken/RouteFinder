import asyncio
import aiohttp
import streamlit as st
import json
import io
import os
from PIL import Image
from pathlib import Path

# ============== CONFIG ==============
AREA_NAME = "ontario_south_bouldering_and_rock"
DATA_PATH = Path(f"src/data/trees/{AREA_NAME}_tree.json")
TAGGED_PATH = Path(f"src/data/tagged_trees/tagged_{AREA_NAME}_tree_test.json")
CACHE_DIR = Path("src/data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PREFETCH_LIMIT = 10  # Number of images to load ahead
# ===================================

# -------- Recursive Image Extractor --------
def extract_routes(data):
    """Recursively extract route images and metadata from hierarchical data."""
    routes = []
    if isinstance(data, dict):
        if "routes" in data:
            for r in data["routes"]:
                routes.append({
                    "route_name": r.get("name"),
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
    """Fetch image from cache if exists, otherwise download and cache it."""
    filename = CACHE_DIR / (url.split("/")[-1].split("?")[0])
    
    # otherwise, download and save
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            content = await resp.read()
            with open(filename, "wb") as f:
                f.write(content)
            return Image.open(io.BytesIO(content))
    except Exception:
        async with session.get(url.replace("large", "medium")) as resp:
            resp.raise_for_status()
            content = await resp.read()
            with open(filename, "wb") as f:
                f.write(content)
            return Image.open(io.BytesIO(content))
    finally:
        return None

async def prefetch_cached_images(urls):
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[fetch_image(session, url) for url in urls])

# Load cached images:
def load_image(url):
    filename = CACHE_DIR / (url.split("/")[-1].split("?")[0])
    # if cached, load from disk
    if filename.exists():
        try:
            return Image.open(filename)
        except Exception:
            filename.unlink(missing_ok=True)  # delete corrupted cache

# Clearing Cached Images:
def remove_image_from_cache(image_path):
    """
    Clears all image files from the specified folder.
    Supported image extensions include .jpg, .jpeg, .png, .gif, .bmp, .tiff.
    """

    if not os.path.isfile(image_path):
        raise(f"Error: Image '{image_path}' not found.")

    try:
        os.remove(image_path)
    except OSError as e:
        print(f"Error removing image: {e}")

@st.cache_data
def load_data(DATA_PATH):
    # Load data
    data = json.load(open(DATA_PATH))
    routes = extract_routes(data)
    all_images = [
        {"route": r["route_name"], "url": img}
        for r in routes for img in r["images"]
    ]
    return all_images

# -------- Main Streamlit App --------
st.title("🧗 Route Image Tagger")
all_images = load_data(DATA_PATH)
st.text("""Rules:
1. Discard photos where the climber or person blocks a significant portion of the route/boulder.
2. Discard topo images, maps, annotated photos, close-ups of bolts, anchors, or scenery not showing the route/boulder.
3. Keep if the full line/boulder of the climb is visible from base to top (even if a rope or a small climber is visible).
4. Prefer front-on ground perspectives; viewing the route/boulder as a climber would before they begin.
        """)

# Load existing tags
tagged = json.load(open(TAGGED_PATH)) if TAGGED_PATH.exists() else {}
untagged = [img for img in all_images if img["url"] not in tagged]

st.write(f"**Progress:** {len(tagged)} tagged / {len(all_images)} total")

if not untagged:
    st.success("🎉 All images tagged!")
    st.text("Clearing Image Cache...")
    for image_path in list(CACHE_DIR.glob("*.jpg")):
        remove_image_from_cache(str(image_path))
    st.stop()

# Get current image
idx = st.session_state.get("index", 0)
if idx >= len(untagged):
    idx = 0

current = untagged[0]
# print(f"CURRENT: {current['url']}")
# Retrieve the number of images currently cached
n_cached_images = len(list(CACHE_DIR.glob("*.jpg")))

# Check if we need to fetch images:
if n_cached_images == 0: # This asks if we've already run out of cached imagesd
    next_urls = [img["url"] for img in untagged[0:PREFETCH_LIMIT]]
    print(next_urls)
    # [print(F"CACHING: {url}") for url in next_urls]
    _ = asyncio.run(prefetch_cached_images(next_urls))

current_img = load_image(current["url"])
print(current["url"])
image_cache_location = CACHE_DIR / (current["url"].split("/")[-1].split("?")[0])

if current_img:
    st.image(current_img, caption=f"{current['route']}", width=500, )
else:
    st.warning("⚠️ Could not load image.")
    if st.button("Skip"):
        tagged[current["url"]] = {
            "route": current["route"],
            "tag": "skipped"
        }
        with open(TAGGED_PATH, "w") as f:
            json.dump(tagged, f, indent=2)
        # remove the cached image
        remove_image_from_cache(image_cache_location)
        st.session_state["index"] = idx + 1
        st.rerun()

    if st.button("Refresh Rerun", use_container_width = True):
        # for image_path in list(CACHE_DIR.glob("*.jpg")):
        #     remove_image_from_cache(str(image_path))
        st.rerun()
    # st.rerun()

# --- ACTION BUTTONS ---
col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    if st.button("✅ Keep", use_container_width=True):
        tagged[current["url"]] = {
            "route": current["route"],
            "tag": "keep"
        }
        with open(TAGGED_PATH, "w") as f:
            json.dump(tagged, f, indent=2)
        # remove the cached image
        remove_image_from_cache(image_cache_location)
        st.session_state["index"] = idx + 1
        st.rerun()

with col3:
    if st.button("❌ Discard", use_container_width=True):
        tagged[current["url"]] = {
            "route": current["route"],
            "tag": "discard"
        }
        with open(TAGGED_PATH, "w") as f:
            json.dump(tagged, f, indent=2)
        # remove the cached image
        remove_image_from_cache(image_cache_location)
        st.session_state["index"] = idx + 1
        st.rerun()


# Note: Keep fearless warrior 1*
