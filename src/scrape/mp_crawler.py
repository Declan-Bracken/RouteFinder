import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
from mp_route_data_collector import get_mountainproject_route_data


# def fetch_routes_selenium(url):
#     """
#     Given the url for a crag from mountainproject, collect the urls for all available routes.
#     """
#     options = Options()
#     options.add_argument('--headless')
#     driver = webdriver.Chrome(options=options)
    
#     driver.get(url)
#     time.sleep(1)  # wait for JS to load
    
#     soup = BeautifulSoup(driver.page_source, 'html.parser')
#     routes = soup.select('tr.route-row')
    
#     route_links = []
#     for route in routes:
#         a_tag = route.select_one('a[href*="/route/"]')
#         if a_tag:
#             route_links.append(a_tag['href'])
    
#     driver.quit()
#     return route_links

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
    # if this is a greater area, return a list of sub areas
    elif area_table:
        # Grab all route ids:
        links = area_table.select("a[href]")
        areas = [a.get('href') for a in links]
        return areas, []
    # edge case
    else:
        return [], []

def dfs_mountain_project(area_url, visited = None):
    """
    Recursively traverse Mountain Project area pages to build an area/route tree.
    """

    if visited is None:
        visited = set()

    # Avoid duplicates / infinite loops, and ensure that we haven't accidentally moved to a different kind of page like a map or profile.
    if area_url in visited or (("/area/" not in area_url) and ("/route/" not in area_url)):
        return {}
    visited.add(area_url)

    # be gentle to the server
    time.sleep(1)

    html = requests.get(area_url).text
    soup = BeautifulSoup(html, 'html.parser')

    node = {
        "type": "area",
        "url": area_url,
        "subareas": {},
        "routes": []
    }

    subareas, routes = fetch_sidebar_links(soup)

    # Recurse for each subarea
    for sub_url in subareas:
        node["subareas"][sub_url] = dfs_mountain_project(sub_url, visited)

    # Collect route data
    for route_url in routes:
        try:
            print(f"  └── Scraping route: {route_url}")
            route_data = get_mountainproject_route_data(route_url)
            node["routes"].append(route_data)
            time.sleep(1)  # rate limit
        except Exception as e:
            print(f"  ⚠️ Failed to scrape {route_url}: {e}")

    return node

def get_tree_from_sitemaps(sitemap_url, visited = None):
    """
    For a sitemap xml url, get all area sitemaps.
    """
    if visited is None:
        visited = set()
    
    # avoid re-fetching the same sitemap:
    if sitemap_url in visited:
        return {}
    visited.add(sitemap_url)

    # When the recursive input is no longer an xml but rather a page, we can begin the true search
    if not sitemap_url.endswith('.xml'):
        tree = dfs_mountain_project(sitemap_url)  # your inner HTML DFS
        return {sitemap_url: tree}
    
    resp = requests.get(sitemap_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'xml')

    # Findall sitemap xmls which contain the 'area' specifier
    area_sitemaps = [sitemap.text for sitemap in soup.find_all('loc') if 'area' in sitemap]

    all_trees = {}

    for area_xml in area_sitemaps:
        # recursive call
        sub_tree = get_tree_from_sitemaps(area_xml, visited)
        # merge results
        all_trees.update(sub_tree)

    return all_trees

def print_nicely(iterable):
    for item in iterable:
        print(item, "\n")


if __name__ == "__main__":
    # master_sitemap_url = "https://www.mountainproject.com/sitemap.xml"
    # test_area_url = "https://www.mountainproject.com/area/105746283/the-needle"
    test_greater_area = "https://www.mountainproject.com/area/105714282/spearfish-canyon"
    # test_leaf_area = "https://www.mountainproject.com/area/105865091/big-picture-wall"
    # urls = fetch_routes_selenium(test_area)
    # print(f"found {len(urls)} routes")
    # print(urls)
    tree = dfs_mountain_project(test_greater_area)
    print(tree)
    # html = requests.get(test_greater_area).text
    # soup = BeautifulSoup(html, 'html.parser')
    # areas, routes = fetch_sidebar_links(soup)
    # print("---------------AREAS------------------\n")
    # print_nicely(areas)
    # print("---------------ROUTES------------------\n")
    # print_nicely(routes)
