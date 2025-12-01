import gzip
import json

gz_path = "data/trees/mountain_project_tree.json.gz"

# Load only the top layer
with gzip.open(gz_path, mode="rt", encoding="utf-8") as f:
    data = json.load(f)

def peek_structure(obj, max_depth=2, depth=0):
    if depth >= max_depth:
        return
    if isinstance(obj, dict):
        print("  " * depth + f"Dict keys: {list(obj.keys())}")
        for k, v in obj.items():
            peek_structure(v, max_depth, depth + 1)
    elif isinstance(obj, list):
        print("  " * depth + f"List of length {len(obj)}")
        for i, item in enumerate(obj[:3]):  # peek at first 3 items only
            peek_structure(item, max_depth, depth + 1)
    else:
        print("  " * depth + f"{type(obj).__name__}: {obj}")

peek_structure(data, max_depth=2)
