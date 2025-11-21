import requests
from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
import re

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

def safe_call(fn, *args, default=None, **kwargs):
    """
    Calls `fn` with given arguments, returns `default` if an exception occurs.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def get_mountainproject_route_data(soup):
    route_name = safe_call(lambda: soup.select("h1")[0].text.strip())
    route_grade = safe_call(lambda: soup.select(".rateYDS")[0].text.strip())

    details_table = safe_call(lambda: soup.select(".description-details")[0])
    route_details = safe_call(collect_route_details, details_table, default={})  # return empty dict if fails

    img_containers = safe_call(lambda: soup.select(".img-container.position-relative"), default=[])
    all_imgs = safe_call(collect_images, img_containers, default=[])

    description_table = safe_call(lambda: soup.select(".mt-2.max-height.max-height-md-800.max-height-xs-600"), default=[])
    description_details = safe_call(collect_description_details, description_table, default={})

    # Deduplicate images
    unique_imgs = list(dict.fromkeys(all_imgs or []))

    return {"name": route_name, "grade": route_grade, "images": unique_imgs, **(route_details or {}), **(description_details or {})} #, "description": route_details

if __name__ == "__main__":
    test_route_url = "https://www.mountainproject.com/route/108221353/the-camel"
    test_route_url_2 = "https://www.mountainproject.com/route/105762594/grizzly-couloir"

    html = requests.get(test_route_url_2).text
    print(html)
    soup = BeautifulSoup(html, parser = "html.parser")
    route_data = get_mountainproject_route_data(soup)
    for key, value in route_data.items():
        print(key.upper())
        print(value, "\n")

    # Okay now let's try to make the same thing for the crag
