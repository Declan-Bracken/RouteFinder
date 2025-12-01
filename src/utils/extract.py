# -------- Recursive Image Extractor --------
# Built for flattening the hierarchical tree
# Used for original streamlit interface.
def extract_routes(data):
    """Recursively extract route images and metadata from hierarchical data."""
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

# def extract_with_lineage(data, lineage = None):
#     """Recursively extract route images and metadata from hierarchical data.
#     Use lineage to route for identification"""
#     routes = []
#     lineage = lineage or []

#     if isinstance(data, dict):
#         print(data.keys())
#         current_area = data.get("url").split("/")[-1]
#         if current_area:
#             lineage = lineage + [current_area]
#         # Chck for routes
#         if "routes" in data:
#             for r in data["routes"]: # data["routes"] is a list of dictionaries where each dictionary contains route info.
#                 route_lineage = " / ".join(lineage)
#                 routes.append({
#                     "route_name": r.get("name"),
#                     "route_lineage": route_lineage,
#                     "images": r.get("images", [])
#                 })
#         for v in data.values():
#             if isinstance(v, (dict, list)):
#                 routes.extend(extract_with_lineage(v, lineage))
#     elif isinstance(data, list):
#         for item in data:
#             routes.extend(extract_with_lineage(item, lineage))
#     return routes

def extract_with_lineage(data, lineage=None):
    """
    Recursively extract route images and metadata from hierarchical data.
    lineage: list of parent area names
    """
    routes = []
    lineage = lineage or []

    if isinstance(data, dict):
        # Standard area info
        current_area = data.get("url", "").split("/")[-1]
        if current_area:
            lineage = lineage + [current_area]

        # If there are routes, extract them
        if "routes" in data:
            for r in data["routes"]:
                route_lineage = " / ".join(lineage)
                routes.append({
                    "route_name": r.get("name"),
                    "route_lineage": route_lineage,
                    "images": r.get("images", [])
                })

        # Recurse over subareas
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                routes.extend(extract_with_lineage(v, lineage))
            # Handle case where dict keys are URLs (second layer)
            elif k.startswith("http") and isinstance(v, dict):
                routes.extend(extract_with_lineage(v, lineage))

    elif isinstance(data, list):
        for item in data:
            routes.extend(extract_with_lineage(item, lineage))

    return routes
