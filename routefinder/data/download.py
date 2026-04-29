import asyncio
import aiohttp
import io
import os
from pathlib import Path
from PIL import Image


async def fetch_image(session, url, cache_dir, min_size=None):
    """
    Download an image, optionally enforce a minimum resolution, and save to cache.
    Returns (PIL.Image, url) on success, (None, url) on failure or size rejection.
    """
    filename = Path(cache_dir) / (url.split("/")[-1].split("?")[0])
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            content = await resp.read()
    except Exception:
        try:
            async with session.get(url.replace("large", "medium")) as resp:
                resp.raise_for_status()
                content = await resp.read()
        except Exception:
            return None, url

    try:
        img = Image.open(io.BytesIO(content))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        if min_size and (img.width < min_size[0] or img.height < min_size[1]):
            return None, url
        img.save(filename, format="JPEG", quality=90, optimize=True)
        return img, url
    except Exception:
        return None, url


async def fetch_image_limited(session, url, cache_dir, semaphore, min_size=None):
    async with semaphore:
        return await fetch_image(session, url, cache_dir, min_size)


async def prefetch_cached_images(urls, cache_dir, semaphore, min_size=None):
    """Download a list of image URLs concurrently. Returns list of (PIL.Image, url) for successes."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image_limited(session, url, cache_dir, semaphore, min_size) for url in urls]
        results = await asyncio.gather(*tasks)
    return [r for r in results if r[0] is not None]


def load_image(url, cache_dir):
    """Load a cached image by URL. Returns PIL.Image or None if not cached / corrupted."""
    filename = Path(cache_dir) / (url.split("/")[-1].split("?")[0])
    if filename.exists():
        try:
            return Image.open(filename)
        except Exception:
            filename.unlink(missing_ok=True)
    return None


def remove_image_from_cache(image_path):
    path = str(image_path)
    if not os.path.isfile(path):
        return
    try:
        os.remove(path)
    except OSError as e:
        print(f"Error removing {path}: {e}")
