"""
Batch samplers for hard negative mining in route retrieval training.

In DDP, training samplers (shuffle=True) split routes across ranks so each
GPU processes a disjoint subset. Val/test samplers (shuffle=False) run on
the full dataset on every rank so metrics are comparable across runs.
"""

import random
from collections import defaultdict
from torch.utils.data import BatchSampler


def group_by_route(hf_dataset) -> list[list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, sample in enumerate(hf_dataset):
        groups[sample["route_id"]].append(i)
    return list(groups.values())


def create_area_buckets(route_groups, hf_dataset):
    area_of = {group[0]: hf_dataset[group[0]]["area_path"] for group in route_groups}
    sorted_groups = sorted(route_groups, key=lambda g: area_of[g[0]])

    area_buckets: dict[str, list[list[int]]] = defaultdict(list)
    for group in sorted_groups:
        area_buckets[area_of[group[0]]].append(group)

    seen: set[str] = set()
    area_order: list[str] = []
    for group in sorted_groups:
        ap = area_of[group[0]]
        if ap not in seen:
            area_order.append(ap)
            seen.add(ap)

    return area_order, area_buckets


def create_split(hf_dataset, train_perc: float, val_perc: float):
    assert (train_perc + val_perc) <= 1.0
    assert (train_perc >= 0) and (val_perc >= 0)

    route_groups = group_by_route(hf_dataset)
    area_order, area_buckets = create_area_buckets(route_groups, hf_dataset)

    N_train = int(train_perc * len(hf_dataset))
    N_val = int(val_perc * len(hf_dataset))

    train_indices, val_indices, test_indices = [], [], []
    img_count = 0
    for area in area_order:
        for route in area_buckets[area]:
            if img_count < N_train:
                train_indices.extend(route)
            elif img_count < N_train + N_val:
                val_indices.extend(route)
            else:
                test_indices.extend(route)
            img_count += len(route)

    return (
        hf_dataset.select(train_indices),
        hf_dataset.select(val_indices),
        hf_dataset.select(test_indices),
    )


def _dist_rank_world():
    """Returns (rank, world_size) if torch.distributed is initialised, else (0, 1)."""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class MultiRouteBatchSampler(BatchSampler):
    """
    Packs complete routes into batches so every SupCon anchor has positives.

    In DDP with shuffle=True (training), routes are split interleaved by rank
    so each GPU owns a disjoint ~half of the routes. With shuffle=False (val/test)
    every rank sees the full dataset so metrics are consistent across runs.
    """
    def __init__(self, hf_dataset, max_batch_size, shuffle=True):
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.route_groups = group_by_route(hf_dataset)

    def _get_groups(self):
        rank, world_size = _dist_rank_world()
        groups = list(self.route_groups)
        if self.shuffle:
            random.shuffle(groups)
            if world_size > 1:
                groups = groups[rank::world_size]
        return groups

    def __iter__(self):
        batch, batch_len = [], 0
        for group in self._get_groups():
            if batch_len + len(group) > self.max_batch_size and batch:
                yield batch
                batch, batch_len = [], 0
            batch.extend(group)
            batch_len += len(group)
        if batch:
            yield batch

    def __len__(self):
        rank, world_size = _dist_rank_world()
        groups = self.route_groups[rank::world_size] if (self.shuffle and world_size > 1) else self.route_groups
        total = sum(len(g) for g in groups)
        return max(1, (total + self.max_batch_size - 1) // self.max_batch_size)


class HardNegativeBatchSampler(MultiRouteBatchSampler):
    """
    Packs batches in DFS area order so same-crag routes land together.
    Rank-split applied after area ordering so locality is preserved per rank.
    """
    def __init__(self, hf_dataset, max_batch_size: int, shuffle: bool = True):
        super().__init__(hf_dataset, max_batch_size, shuffle)
        self._area_order, self._area_buckets = create_area_buckets(self.route_groups, hf_dataset)

    def _get_groups(self):
        rank, world_size = _dist_rank_world()
        ordered = []
        for area in self._area_order:
            routes = list(self._area_buckets[area])
            if self.shuffle:
                random.shuffle(routes)
            ordered.extend(routes)
        if self.shuffle and world_size > 1:
            ordered = ordered[rank::world_size]
        return ordered
