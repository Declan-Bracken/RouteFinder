import re
import requests
from bs4 import BeautifulSoup


def normalize_img_url(src):
    src = src.replace("_topo", "")
    src = re.sub(r'_(smallMed|small|medium)', '_large', src)
    return src


def collect_route_details(details_table):
    route_details = {}
    for row in details_table.find_all("tr"):
        elements = row.find_all("td")
        route_details[elements[0].text.strip()] = elements[1].text.strip()
    return route_details


def collect_images(img_containers):
    all_imgs = []
    for container in img_containers:
        img_tag = container.select_one("img")
        if img_tag:
            src = img_tag.get("data-src")
            if src:
                src = normalize_img_url(src)
                all_imgs.append(src)
    return all_imgs


def collect_description_details(description_table):
    description_details = {}
    for description in description_table:
        title = description.select(".mt-2")
        desc = description.select(".fr-view")
        description_details[title[0].text.strip()] = desc[0].text.strip()
    return description_details


def safe_call(fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def get_mountainproject_route_data(soup):
    route_name = safe_call(lambda: soup.select("h1")[0].text.strip())
    route_grade = safe_call(lambda: soup.select(".rateYDS")[0].text.strip())

    details_table = safe_call(lambda: soup.select(".description-details")[0])
    route_details = safe_call(collect_route_details, details_table, default={})

    img_containers = safe_call(lambda: soup.select(".img-container.position-relative"), default=[])
    all_imgs = safe_call(collect_images, img_containers, default=[])

    description_table = safe_call(
        lambda: soup.select(".mt-2.max-height.max-height-md-800.max-height-xs-600"), default=[]
    )
    description_details = safe_call(collect_description_details, description_table, default={})

    unique_imgs = list(dict.fromkeys(all_imgs or []))
    return {
        "name": route_name,
        "grade": route_grade,
        "images": unique_imgs,
        **(route_details or {}),
        **(description_details or {}),
    }
