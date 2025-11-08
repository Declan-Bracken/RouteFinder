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

def get_mountainproject_route_data(soup):
    # Soup is a BeautifulSoup4 Html tree object
    route_name = soup.select("h1")[0].text.strip()
    route_grade = soup.select(".rateYDS")[0].text.strip()
    # Get route details (type, length, FA, site views)
    details_table = soup.select(".description-details")[0]
    route_details = collect_route_details(details_table)
    # Get images
    img_containers = soup.select(".img-container.position-relative")
    all_imgs = collect_images(img_containers)
    # Get description details (sentences describing route or necessary protection)
    description_table= soup.select(".mt-2.max-height.max-height-md-800.max-height-xs-600")
    description_details = collect_description_details(description_table)

    # Deduplicate while preserving order
    unique_imgs = list(dict.fromkeys(all_imgs))

    return {"name": route_name, "grade": route_grade, "images": unique_imgs, **route_details, **description_details} #, "description": route_details

if __name__ == "__main__":
    test_route_url = "https://www.mountainproject.com/route/108221353/the-camel"
    test_route_url_2 = "https://www.mountainproject.com/route/108221381/the-big-f"
    html = requests.get(test_route_url_2).text
    print(html)
    soup = BeautifulSoup(html, parser = "html.parser")
    route_data = get_mountainproject_route_data(soup)
    for key, value in route_data.items():
        print(key.upper())
        print(value, "\n")
