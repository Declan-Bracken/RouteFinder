"""
Scrape Mountain Project and write a gzipped JSON area/route tree.
Usage: python scripts/scrape.py
"""
import asyncio
import aiohttp
import gzip
import json
import logging
import os
import time
from bs4 import BeautifulSoup
import requests

from routefinder.data.scrape import get_mountainproject_route_data

SEM = asyncio.Semaphore(10)
COUNTER_LOCK = asyncio.Lock()

SITEMAPS_VISITED = 0
AREAS_VISITED = 0
ROUTES_COLLECTED = 0
IMAGES_COLLECTED = 0
FAILED_URLS = []

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("scraper.log")],
)


def fetch_sidebar_links(soup):
    route_table = soup.find(id="left-nav-route-table")
    area_table = soup.find(class_="max-height max-height-md-0 max-height-xs-400")
    if route_table:
        return [], [a.get("href") for a in route_table.select("a[href]")]
    elif area_table:
        return [a.get("href") for a in area_table.select("a[href]")], []
    return [], []


async def safe_fetch(session, url):
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
            if not soup.select("h1"):
                raise ValueError("No <h1> found, possible error page")
            return html
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        FAILED_URLS.append({"url": url, "error": str(e)})
        return None


async def scrape_route(session, url):
    global ROUTES_COLLECTED, IMAGES_COLLECTED
    html = await safe_fetch(session, url)
    if html is None:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    logging.info(f"[ROUTE #{ROUTES_COLLECTED}] {url}")
    route_data = get_mountainproject_route_data(soup)
    route_data["url"] = url
    async with COUNTER_LOCK:
        ROUTES_COLLECTED += 1
        IMAGES_COLLECTED += len(route_data["images"])
    return route_data


async def dfs_area(session, area_url, visited):
    global AREAS_VISITED
    if area_url in visited or (("/area/" not in area_url) and ("/route/" not in area_url)):
        return {}
    visited.add(area_url)
    async with COUNTER_LOCK:
        AREAS_VISITED += 1
    logging.info(f"[AREA #{AREAS_VISITED}] {area_url}")

    html = await safe_fetch(session, area_url)
    if html is None:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    subareas, routes = fetch_sidebar_links(soup)

    node = {"type": "area", "url": area_url, "subareas": {}, "routes": []}
    route_data_list = await asyncio.gather(*[scrape_route(session, r) for r in routes])
    node["routes"].extend(route_data_list)
    for sub_url in subareas:
        node["subareas"][sub_url] = await dfs_area(session, sub_url, visited)
    return node


async def dfs_area_with_session(url, visited):
    async with aiohttp.ClientSession() as session:
        return await dfs_area(session, url, visited)


def get_tree_from_sitemaps(sitemap_url, visited):
    global SITEMAPS_VISITED
    if not sitemap_url.endswith(".xml"):
        return asyncio.run(dfs_area_with_session(sitemap_url, visited))
    if sitemap_url in visited:
        return {}
    visited.add(sitemap_url)
    SITEMAPS_VISITED += 1
    logging.info(f"[SITEMAP #{SITEMAPS_VISITED}] {sitemap_url}")

    try:
        resp = requests.get(sitemap_url)
        resp.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to fetch {sitemap_url}: {e}")
        FAILED_URLS.append({"url": sitemap_url, "error": str(e)})
        return None

    soup = BeautifulSoup(resp.text, "xml")
    area_sitemaps = [s.text for s in soup.find_all("loc") if "area" in s.text]
    node = {"type": "sitemap", "url": sitemap_url, "subareas": {}}
    for area_xml in area_sitemaps:
        sub_tree = get_tree_from_sitemaps(area_xml, visited)
        if sub_tree:
            node["subareas"][sub_tree["url"]] = sub_tree
    return node


if __name__ == "__main__":
    master_sitemap_url = "https://www.mountainproject.com/sitemap.xml"
    area_name = "mountain_project"
    visited = set()

    start_time = time.time()
    tree = get_tree_from_sitemaps(master_sitemap_url, visited)
    end_time = time.time()

    out_dir = "data/trees"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{area_name}_tree.json.gz")

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(tree, f)
    logging.info(f"Tree saved to {out_path}")

    if FAILED_URLS:
        logging.info(f"Total failed URLs: {len(FAILED_URLS)}")
    logging.info(f"Time: {end_time - start_time:.2f}s | Sitemaps: {SITEMAPS_VISITED} | Areas: {AREAS_VISITED} | Routes: {ROUTES_COLLECTED} | Images: {IMAGES_COLLECTED}")
