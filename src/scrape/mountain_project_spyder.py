import requests
from bs4 import BeautifulSoup
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re

BASE_URL = "https://www.mountainproject.com"
SAVE_DIR = "data/raw"

def fetch_routes_selenium(url):
    """
    Given the url for a crag from mountainproject, collect the urls for all available routes.
    """
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    
    driver.get(url)
    time.sleep(1)  # wait for JS to load
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    routes = soup.select('tr.route-row')
    
    route_links = []
    for route in routes:
        a_tag = route.select_one('a[href*="/route/"]')
        if a_tag:
            route_links.append(a_tag['href'])
    
    driver.quit()
    return route_links

def normalize_img_url(src):
    # Remove topo
    src = src.replace("_topo", "")
    # Replace any size indicator (_small, _smallMed, _medium, etc.) with _large
    src = re.sub(r'_(smallMed|small|medium)', '_large', src)
    return src

def collect_route_details(details_table):
    route_details = {}
    # Search through each row of the table
    for row in details_table.find_all("tr"):
        # Search through each element of the current row
        elements = row.find_all("td")
        route_details[elements[0].text.strip()] = elements[1].text.strip()
    return route_details

def collect_images(img_containers):
    all_imgs = []
    # Get urls for all images found
    for container in img_containers:
        img_tag = container.select_one("img")
        if img_tag:
            src = img_tag.get("data-src")
            if src:
                # fix sizing and remove topo
                src = normalize_img_url(src)
                all_imgs.append(src)
    return all_imgs

def collect_description_details(description_table):
    
    description_details = {}
    for description in description_table:
        title = description.select(".mt-2")
        description = description.select(".fr-view")
        description_details[title[0].text.strip()] = description[0].text.strip()
    return description_details

def get_mountainproject_route_data(route_url):
    html = requests.get(route_url).text
    soup = BeautifulSoup(html, 'html.parser')

    route_name = soup.select("h1")[0].text.strip()
    route_grade = soup.select(".rateYDS")[0].text.strip()
    # Get route details
    details_table = soup.select(".description-details")[0]
    route_details = collect_route_details(details_table)
    # Get images
    img_containers = soup.select(".img-container.position-relative")
    all_imgs = collect_images(img_containers)
    # Get description details
    description_table= soup.select(".mt-2.max-height.max-height-md-800.max-height-xs-600")
    description_details = collect_description_details(description_table)

    # Deduplicate while preserving order
    unique_imgs = list(dict.fromkeys(all_imgs))

    return {"name": route_name, "grade": route_grade, "images": unique_imgs, "url": route_url, **route_details, **description_details} #, "description": route_details

if __name__ == "__main__":
    area_url = "https://www.mountainproject.com/area/classics/106358863/mount-nemo"
    # test_route_url = "https://www.mountainproject.com/route/108221353/the-camel"
    route_links = fetch_routes_selenium(area_url)

    os.makedirs(SAVE_DIR, exist_ok=True)
    for i, link in enumerate(route_links):
        try:
            data = get_route_data(link)
            print(f"[{i+1}/{len(route_links)}] {data['name']}")
            # save JSON line per route
            with open(f"{SAVE_DIR}/routes.jsonl", "a") as f:
                f.write(str(data) + "\n")
        except Exception as e:
            print("Error on", link, e)
        break
