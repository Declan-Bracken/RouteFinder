import requests
from bs4 import BeautifulSoup
import json, gzip
import time
from mp_route_data_collector import get_mountainproject_route_data
# Asynchronous programming
import asyncio
import aiohttp
import logging
import os

# Global shared concurrency control
SEM = asyncio.Semaphore(10)
COUNTER_LOCK = asyncio.Lock()

# Shared counters
SITEMAPS_VISITED = 0
AREAS_VISITED = 0
ROUTES_COLLECTED = 0
IMAGES_COLLECTED = 0
FAILED_URLS = []

# logging:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),               # Console output
        logging.FileHandler("scraper.log")     # Also save to file
    ]
)

def fetch_sidebar_links(soup):
    # Check if there are route table elements
    route_table = soup.find(id="left-nav-route-table")
    area_table = soup.find(class_="max-height max-height-md-0 max-height-xs-400")

    # If this is a leaf area, return a list of routes
    if route_table:
        # Grab all route ids:
        links = route_table.select("a[href]")
        routes = [a.get('href') for a in links]
        return [], routes
    # if this is a greater area, retsurn a list of sub areas
    elif area_table:
        # Grab all route ids:
        links = area_table.select("a[href]")
        areas = [a.get('href') for a in links]
        return areas, []
    # edge case
    else:
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
        logging.error(f"Failed to fetch {url} after 1 attempts")
        FAILED_URLS.append({"url": url, "error": str(e)})
        return None

async def scrape_route(session, url):
    global ROUTES_COLLECTED
    global IMAGES_COLLECTED

    html = await safe_fetch(session, url)
    if html is None:
        return {}
    soup = BeautifulSoup(html, "html.parser")

    logging.info(f"[ROUTE #{ROUTES_COLLECTED}] Visited {url}")
    route_data = get_mountainproject_route_data(soup)
    route_data["url"] = url
    logging.info(f"N-Images: {len(route_data['images'])}")

    # Increment counter
    async with COUNTER_LOCK:
         ROUTES_COLLECTED += 1
         IMAGES_COLLECTED += len(route_data['images'])

    return route_data

async def dfs_area(session, area_url, tree, visited):
    """
    Recursively traverse Mountain Project area pages to build an area/route tree.
    """
    global AREAS_VISITED

    # Avoid duplicates / infinite loops, and ensure that we haven't accidentally moved to a different kind of page like a map or profile.
    if area_url in visited or (("/area/" not in area_url) and ("/route/" not in area_url)):
        return {}
    visited.add(area_url)

    # increment counter
    async with COUNTER_LOCK:
        AREAS_VISITED += 1

    logging.info(f"[AREA #{AREAS_VISITED}] Visited {area_url}")

    # Get area page asynchronously, if a server error occurs, continue.
    html = await safe_fetch(session, area_url)
    if html is None:
        return {}
    soup = BeautifulSoup(html, 'html.parser')

    # get the list of route/area urls
    subareas, routes = fetch_sidebar_links(soup)

    node = {
        "type": "area",
        "url": area_url,
        "subareas": {},
        "routes": []
    }

    # Collect route data
    tasks = [scrape_route(session, r) for r in routes]
    route_data_list = await asyncio.gather(*tasks)
    node["routes"].extend(route_data_list)

    # Recurse for each subarea (await response)
    for sub_url in subareas:
        node["subareas"][sub_url] = await dfs_area(session, sub_url, tree, visited)

    return node

async def dfs_area_with_session(url, visited):
    async with aiohttp.ClientSession() as session:
        return await dfs_area(session, url, {}, visited)

def get_tree_from_sitemaps(sitemap_url, visited):
    """
    For a sitemap xml url, get all area sitemaps.
    """
    global SITEMAPS_VISITED
    
    # When the recursive input is no longer an xml but rather a page, we can begin the true search. We have a visited check in dfs_area, don't worry.
    if not sitemap_url.endswith('.xml'):
        node = asyncio.run(dfs_area_with_session(sitemap_url, visited))  # your inner HTML DFS
        return node
    
    # avoid re-fetching the same sitemap:
    if sitemap_url in visited:
        return {}
    visited.add(sitemap_url)
    
    SITEMAPS_VISITED += 1
    logging.info(f"[SITEMAP #{SITEMAPS_VISITED}] Visited {sitemap_url}")

    # Error catching
    try:
        resp = requests.get(sitemap_url)
        resp.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to fetch {sitemap_url} after 1 attempts")
        FAILED_URLS.append({"url": sitemap_url, "error": str(e)})
        return None
    
    soup = BeautifulSoup(resp.text, 'xml')
    # Findall sitemap xmls which contain the 'area' specifier
    area_sitemaps = [sitemap.text for sitemap in soup.find_all('loc') if 'area' in sitemap.text]

    node = {
        "type": "sitemap",
        "url": sitemap_url,
        "subareas": {},
        # no need to include routes
    }

    for area_xml in area_sitemaps:
        # recursive call
        sub_tree = get_tree_from_sitemaps(area_xml, visited)
        # merge results
        if sub_tree:
            node["subareas"][sub_tree["url"]] = sub_tree

    return node

def print_nicely(iterable):
    for item in iterable:
        print(item, "\n")


if __name__ == "__main__":
    
    # url = "https://www.mountainproject.com/area/106358845/golden-horseshoe"
    # url = "https://www.mountainproject.com/area/118721316/ontario-south-bouldering-and-rock"
    # area_name = url.split("/")[-1].replace("-","_")
    # tree = asyncio.run(dfs_area_with_session(url, visited))

    master_sitemap_url = "https://www.mountainproject.com/sitemap.xml"
    area_name = "mountain_project"
    visited = set()
    
    start_time = time.time()
    tree = get_tree_from_sitemaps(master_sitemap_url, visited)
    end_time = time.time()

    # Your target path
    directory = "src/data/trees"
    filename_gz = f"{area_name}_tree.json.gz"
    filepath_gz = os.path.join(directory, filename_gz)

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    # Save as JSON

    try:
        with gzip.open(filepath_gz, "wt", encoding="utf-8") as f:
            json.dump(tree, f)
        logging.info(f"Scraping complete. Tree saved to {directory}")
    except Exception as e:
        logging.error(f"Gzip failed → keeping raw json only. Error: {e}")

    if FAILED_URLS:
        logging.info("\n----- SCRAPER ERROR REPORT -----")
        # for err in FAILED_URLS:
        #     logging.info(f"{err['url']} -> {err['error']}")
        logging.info(f"Total failed URLs: {len(FAILED_URLS)}")

    logging.info(f"Program Time: {end_time - start_time:.2f} seconds")
    logging.info(f"# of Site Maps Visisted {SITEMAPS_VISITED}")
    logging.info(f"# of Areas Visisted {AREAS_VISITED}")
    logging.info(f"# of Routes Collected {ROUTES_COLLECTED}")
    logging.info(f"# of Images Collected {IMAGES_COLLECTED}")
