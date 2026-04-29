import gzip
import json
from pathlib import Path


def extract_routes(data):
    """Recursively extract route images and metadata from hierarchical tree data."""
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


def extract_with_lineage(data, lineage=None):
    """Recursively extract routes with their area lineage from hierarchical tree data."""
    routes = []
    lineage = lineage or []

    if isinstance(data, dict):
        current_area = data.get("url", "").split("/")[-1]
        if current_area:
            lineage = lineage + [current_area]

        if "routes" in data:
            for r in data["routes"]:
                routes.append({
                    "route_name": r.get("name"),
                    "route_lineage": " / ".join(lineage),
                    "images": r.get("images", [])
                })

        for k, v in data.items():
            if isinstance(v, (dict, list)):
                routes.extend(extract_with_lineage(v, lineage))
            elif k.startswith("http") and isinstance(v, dict):
                routes.extend(extract_with_lineage(v, lineage))

    elif isinstance(data, list):
        for item in data:
            routes.extend(extract_with_lineage(item, lineage))

    return routes


def load_flattened_tree(file_path):
    """Load a gzipped JSON tree and return a flat list of route records."""
    with gzip.open(file_path, mode="rt", encoding="utf-8") as f:
        data = json.load(f)
    return extract_with_lineage(data)
