import asyncio, aiohttp
import io
import os
from PIL import Image

# -------- Async Image Prefetching --------
async def fetch_image_limited(session, url, cache_dir, semaphore): # Wrap requests in a semaphore
    async with semaphore:
        return await fetch_image(session, url, cache_dir)
    
async def fetch_image(session, url, cache_dir):
    """Fetch image from cache if exists, otherwise download and cache it."""
    filename = cache_dir / (url.split("/")[-1].split("?")[0])
    
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

async def prefetch_cached_images(urls, cache_dir, semaphore):
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[fetch_image_limited(session, url, cache_dir, semaphore) for url in urls])

# Load cached images:
def load_image(url, cache_dir):
    filename = cache_dir / (url.split("/")[-1].split("?")[0])
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

    assert os.path.isfile(image_path), f"Error: Image not found {str(image_path)}."

    try:
        os.remove(image_path)
    except OSError as e:
        print(f"Error removing image: {e}")
