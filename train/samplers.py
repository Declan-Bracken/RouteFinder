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


class HardNegMiningBatchSampler(MultiRouteBatchSampler):
    """
    Online hard negative mining sampler.

    Falls back to random batching until update_index() is called. Once the
    index is populated, composes each batch around a seed route:
      1. Seed route images
      2. Randomly-sampled routes from the seed's hard negative superset
         (up to hard_neg_frac * max_batch_size images; shuffled each epoch
         so the model doesn't see the same hard pairs every time)
      3. Random routes to fill the remainder (prevents embedding collapse)

    Each route serves as a seed at most once per epoch. Routes absorbed as
    hard negatives or random fill are skipped when their turn as seed arrives,
    so every image appears in exactly one batch per epoch.
    """
    def __init__(self, hf_dataset, max_batch_size: int, hard_neg_frac: float = 0.7, shuffle: bool = True):
        super().__init__(hf_dataset, max_batch_size, shuffle)
        self.hard_neg_frac = hard_neg_frac
        self._hard_neg_index: dict[int, list[int]] | None = None
        self._route_id_to_group: dict[int, list[int]] = {
            hf_dataset[group[0]]["route_id"]: group
            for group in self.route_groups
        }
        self._all_route_ids = list(self._route_id_to_group.keys())

    def update_index(self, hard_neg_index: dict[int, list[int]]) -> None:
        """Refresh the hard negative candidate list. Called by HardNegMiningCallback."""
        self._hard_neg_index = hard_neg_index

    def __iter__(self):
        if self._hard_neg_index is None:
            yield from super().__iter__()
            return

        rank, world_size = _dist_rank_world()
        all_ids = list(self._all_route_ids)
        random.shuffle(all_ids)
        if self.shuffle and world_size > 1:
            all_ids = all_ids[rank::world_size]

        # Separate random pool for remainder fill (different shuffle → more variety)
        random_pool = list(all_ids)
        random.shuffle(random_pool)
        rp_idx = 0
        used: set[int] = set()
        hard_target = int(self.hard_neg_frac * self.max_batch_size)

        for seed_id in all_ids:
            if seed_id in used:
                continue
            used.add(seed_id)
            batch = list(self._route_id_to_group[seed_id])

            # Sample hard negatives from superset (shuffled for batch diversity across epochs)
            candidates = list(self._hard_neg_index.get(seed_id, []))
            random.shuffle(candidates)
            for cand_id in candidates:
                if len(batch) >= hard_target:
                    break
                if cand_id in used or cand_id not in self._route_id_to_group:
                    continue
                cand_group = self._route_id_to_group[cand_id]
                if len(batch) + len(cand_group) > self.max_batch_size:
                    continue
                batch.extend(cand_group)
                used.add(cand_id)

            # Fill remaining slots with random unused routes
            while len(batch) < self.max_batch_size and rp_idx < len(random_pool):
                rid = random_pool[rp_idx]
                rp_idx += 1
                if rid in used or rid not in self._route_id_to_group:
                    continue
                group = self._route_id_to_group[rid]
                if len(batch) + len(group) > self.max_batch_size:
                    continue
                batch.extend(group)
                used.add(rid)

            yield batch
