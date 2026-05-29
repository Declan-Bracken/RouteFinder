"""
Batch samplers for hard negative mining in route retrieval training.

HardNegativeBatchSampler sorts route groups by area_path lexicographically,
which is equivalent to a DFS traversal of the area tree (because the path
encodes the full ancestry — lex order groups all paths under the same prefix
together and places siblings adjacent to each other).

The result: same-crag routes always land in the same batch, and when a batch
needs more routes to fill it, it pulls from the geographically nearest crag
in the tree first.

Curriculum training schedule:
    Stage 1: use MultiRouteBatchSampler (random batches, easy negatives)
    Stage 2: use HardNegativeBatchSampler (tree-proximity batches, hard negatives)

Example:
    # Stage 1
    sampler = MultiRouteBatchSampler(hf_train, max_batch_size=cfg.batch_size)

    # Stage 2 — swap sampler, no other changes
    sampler = HardNegativeBatchSampler(hf_train, max_batch_size=cfg.batch_size)
    loader  = DataLoader(dataset, batch_sampler=sampler, num_workers=cfg.num_workers)
"""

import random
from collections import defaultdict
from torch.utils.data import BatchSampler

def group_by_route(hf_dataset) -> list[list[int]]:
    """Returns list-of-lists where each inner list holds dataset indices for one route."""
    groups: dict[int, list[int]] = defaultdict(list)
    for i, sample in enumerate(hf_dataset):
        groups[sample["route_id"]].append(i)
    return list(groups.values()) # [[hf_ds_row1, hf_ds_row_2], [hf_ds_row3,...], ...]

def create_area_buckets(route_groups, hf_dataset) -> tuple[list[str], dict[str, list[list[int]]]]:
    # Cache area_path for each route group (keyed by first index in the group)
    area_of: dict[int, str] = {
        group[0]: hf_dataset[group[0]]["area_path"]
        for group in route_groups
    }

    # Sort route groups by area_path — this is the DFS ordering
    sorted_groups = sorted(route_groups, key=lambda g:area_of[g[0]])

    # Bucket routes by area_path for locality-preserving shuffle
    area_buckets: dict[str, list[list[int]]] = defaultdict(list)
    for group in sorted_groups:
        area_buckets[area_of[group[0]]].append(group)

    # Area ordering (DFS order, stable)
    seen: set[str] = set()
    area_order: list[str] = []
    for group in sorted_groups:
        ap = area_of[group[0]]
        if ap not in seen:
            area_order.append(ap)
            seen.add(ap)
            
    return (area_order, area_buckets) # return the areas ordered by proximity, and the route groups within each area

def create_split(hf_dataset, train_perc: float, val_perc: float):
    assert (train_perc + val_perc) <= 1.0, "Train and Val percentage must not exceed 1."
    assert (train_perc >= 0) & (val_perc >= 0), "Train and Val percentages must be positive."

    # Always split train/val/test by hard negatives.
    route_groups = group_by_route(hf_dataset)
    area_order, area_buckets = create_area_buckets(route_groups, hf_dataset)

    # Create train, val, test splits by taking contiguous areas. (Assume the image count by area will average out across the sorted dataset)
    N_images = len(hf_dataset)
    N_train = int(train_perc * N_images)
    N_val = int(val_perc * N_images)

    train_indices, val_indices, test_indices = [], [], []
    img_count = 0
    # Each area is a list of routes
    for area in area_order:
        routes = area_buckets[area]
        # Each route is a list of image indices as they appear in hf_dataset
        for route in routes: 
            if img_count < N_train:
                train_indices.extend(route)
            elif (img_count >= N_train) & (img_count < (N_train + N_val)):
                val_indices.extend(route)
            else:
                test_indices.extend(route)
            img_count += len(route)

    train_split = hf_dataset.select(train_indices)
    val_split = hf_dataset.select(val_indices)
    test_split = hf_dataset.select(test_indices)

    return train_split, val_split, test_split

class MultiRouteBatchSampler(BatchSampler):
    """
    Packs several complete routes into each batch. SupCon requires multiple
    samples per class within a batch — without this, most anchors have no
    positives and the loss is meaningless.
    """
    def __init__(self, hf_dataset, max_batch_size, shuffle=True):
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.route_groups = group_by_route(hf_dataset)
    
    def _order_groups(self):
        if self.shuffle:
            random.shuffle(self.route_groups)
        return self.route_groups

    def __iter__(self):
        ordered_groups = self._order_groups()
        batch, batch_len = [], 0
        for group in ordered_groups:
            if batch_len + len(group) > self.max_batch_size and batch:
                yield batch
                batch, batch_len = [], 0
            batch.extend(group)
            batch_len += len(group)
        if batch:
            yield batch

    def __len__(self):
        total = sum(len(g) for g in self.route_groups)
        return max(1, (total + self.max_batch_size - 1) // self.max_batch_size)
    

class HardNegativeBatchSampler(MultiRouteBatchSampler):
    """
    Packs batches in DFS area order (= lex sort of area_path) so that:
      - All images of the same route always land in the same batch.
      - When a batch isn't full after one crag, it pulls from the
        structurally nearest crag in the tree.

    With shuffle=True (default), route order within each crag is randomised
    per epoch, which shifts batch boundaries and provides diversity while
    preserving geographic locality.
    """

    def __init__(self, hf_dataset, max_batch_size: int, shuffle: bool = True):
        super().__init__(hf_dataset, max_batch_size, shuffle)

        self._area_order, self._area_buckets = create_area_buckets(self.route_groups, hf_dataset)

    def _order_groups(self):
        ordered_groups = []
        for area in self._area_order:
            routes = self._area_buckets[area]
            if self.shuffle:
                random.shuffle(routes)
            ordered_groups.extend(routes)
        return ordered_groups
