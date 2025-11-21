import json
from collections import defaultdict
from itertools import combinations

TAGGED_PATH = "src/data/tagged_trees/tagged_ontario_south_bouldering_and_rock_tree.json"

data = json.load(open(TAGGED_PATH))

# Count totals
total = len(data)
kept = sum(1 for _, v in data.items() if v["tag"] == "keep")
discarded = total - kept

# Group kept images by route
route_to_images = defaultdict(list)
for url, meta in data.items():
    if meta["tag"] == "keep":
        route_to_images[meta["route"]].append(url)

# Count multi-view usable routes
multiview_routes = {route: imgs for route, imgs in route_to_images.items() if len(imgs) >= 2}
multiview_total_images = sum(len(imgs) for imgs in multiview_routes.values())

# Count positive pairs
positive_pairs = sum(len(imgs) * (len(imgs) - 1) // 2 for imgs in multiview_routes.values())

print("📌 Dataset Summary")
print("----------------------------")
print(f"Total labeled images: {total}")
print(f"Kept (usable for SimCLR pretrain): {kept}")
print(f"Discarded: {discarded}")
print(f"Routes with ≥2 kept images: {len(multiview_routes)}")
print(f"Total multiview images (belonging to these routes): {multiview_total_images}")
print(f"Positive image pairs available: {positive_pairs}\n")

print("🔍 Top multiview candidates (routes with most images):")
for r, imgs in sorted(multiview_routes.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
    print(f"{r}: {len(imgs)} images")
