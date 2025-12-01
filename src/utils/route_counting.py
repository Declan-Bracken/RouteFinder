import json, gzip
import pandas as pd
from collections import defaultdict
from itertools import combinations
from extract import extract_with_lineage

# TAGGED_PATH = "src/data/tagged_trees/tagged_ontario_south_bouldering_and_rock_tree.json"

# data = json.load(open(TAGGED_PATH))

# # Count totals
# total = len(data)
# kept = sum(1 for _, v in data.items() if v["tag"] == "keep")
# discarded = total - kept

# # Group kept images by route
# route_to_images = defaultdict(list)
# for url, meta in data.items():
#     if meta["tag"] == "keep":
#         route_to_images[meta["route"]].append(url)

# # Count multi-view usable routes
# multiview_routes = {route: imgs for route, imgs in route_to_images.items() if len(imgs) >= 2}
# multiview_total_images = sum(len(imgs) for imgs in multiview_routes.values())

# # Count positive pairs
# positive_pairs = sum(len(imgs) * (len(imgs) - 1) // 2 for imgs in multiview_routes.values())

# print("📌 Dataset Summary")
# print("----------------------------")
# print(f"Total labeled images: {total}")
# print(f"Kept (usable for SimCLR pretrain): {kept}")
# print(f"Discarded: {discarded}")
# print(f"Routes with ≥2 kept images: {len(multiview_routes)}")
# print(f"Total multiview images (belonging to these routes): {multiview_total_images}")
# print(f"Positive image pairs available: {positive_pairs}\n")

# print("🔍 Top multiview candidates (routes with most images):")
# for r, imgs in sorted(multiview_routes.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
#     print(f"{r}: {len(imgs)} images")

# DATA_PATH = "src/data/trees/mountain_project_tree.json.gz"

# with gzip.open(DATA_PATH, mode="rt", encoding="utf-8") as f:
#     data = json.load(f)

# flattened_routes = extract_with_lineage(data)

# img_counts = defaultdict(int)
# for route in flattened_routes:
#     img_count = len(route["images"])
#     img_counts[img_count] += 1

# total_routes = 0
# total_imgs_multiview = 0
# for img_count in sorted(list(img_counts.keys())):
#     route_count = img_counts[img_count]
#     if img_count > 1:
#         total_routes += route_count
#         total_imgs_multiview += route_count * img_count
#     print(f"# Routes with {img_count} images: {route_count}")
# print(f"Total Routes: {total_routes}")
# print(f"Total Multiview Iamges: {total_imgs_multiview}")


# # Compute % of routes with >1 images for all lineages.
# # First compute the total route count and the number of routes with multiple images per lineage
# count_lineages = defaultdict(int) # tuple[0] is route count, tuple[1] is multiview route count
# multiview_count_lineages = defaultdict(int)
# for route in flattened_routes:
#     count_lineages[route["route_lineage"]] += 1 # Add 1 to the route count
#     n_images = len(route["images"])
#     if n_images > 1:
#         multiview_count_lineages[route["route_lineage"]] += 1 # Add 1 to the multiview route count

# # Now compute, for each linage, what percent of their routes are multiview
# percent_multiview = defaultdict(int)
# for route in count_lineages:
#     percent = int(multiview_count_lineages[route] / count_lineages[route] * 100) # compute rounded percentage for simplicity
#     percent_multiview[percent] += 1

# for perc in sorted(percent_multiview.keys()):
#     print(f"# of lineages with {perc}% multiview: {percent_multiview[perc]}")


df = pd.read_csv("src/data/tagged_trees/processed_mountain_project_tree.csv")

n_images = len(df[df["keep"] == True])
n_routes = len(df[df["keep"] == True].groupby("label"))
print(f"Curated {n_images} images spanning {n_routes} routes")
