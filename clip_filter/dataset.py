import asyncio
import hashlib
import io
from pathlib import Path
import numpy as np
import aiohttp
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from torchvision import transforms

ANNOTATED_CSV  = Path("data/tagged_trees/processed_mountain_project_tree.csv")
CACHE_DIR      = Path("data/clip_filter_cache")
CHECKPOINT_DIR = Path("models/clip_filter")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
HEADERS   = {"User-Agent": "Mozilla/5.0 (compatible; RouteFinder-train/1.0)"}

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

class RoutePhotoDataset(Dataset):
    def __init__(self, records: list[tuple[Path, int]], transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)
    

def _cache_path(url: str) -> Path:
    return CACHE_DIR / (hashlib.md5(url.encode()).hexdigest() + ".jpg")

async def _fetch(session, url, sem):
    dest = _cache_path(url)
    if dest.exists():
        return dest
    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=20)) as r:
                    if r.status == 200:
                        data = await r.read()
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        img.save(dest, format="JPEG", quality=90)
                        return dest
                    if r.status in (403, 404, 410):
                        return None
            except Exception:
                pass
            await asyncio.sleep(1.5 ** attempt)
    return None


async def _cache_all(urls, concurrency):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrency)) as session:
        return await asyncio.gather(*[_fetch(session, u, sem) for u in urls])


def download_and_cache(df, concurrency=30):
    print(f"Caching {len(df)} images to {CACHE_DIR}…")
    paths = asyncio.run(_cache_all(df["url"].tolist(), concurrency))
    df = df.copy()
    df["cache_path"] = paths
    before = len(df)
    df = df[df["cache_path"].notna()].reset_index(drop=True)
    print(f"Cached {len(df)} OK, {before - len(df)} failed")
    return df

def make_val_split(df, train_frac):
    """Add a 'split' column using a seeded stratified split. Same seed used everywhere."""
    rng = np.random.default_rng(42)
    df = df.copy()
    df["split"] = "val"
    for keep_val in [True, False]:
        idx = df[df["keep"] == keep_val].index
        n_train = int(len(idx) * train_frac)
        df.loc[rng.choice(idx, size=n_train, replace=False), "split"] = "train"
    return df


def make_loaders(df, train_frac, batch_size):
    df = make_val_split(df, train_frac)

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]

    train_records = [(Path(r.cache_path), int(r.keep)) for r in train_df.itertuples()]
    val_records   = [(Path(r.cache_path), int(r.keep)) for r in val_df.itertuples()]

    # Weighted sampler to balance classes during training
    labels = [r[1] for r in train_records]
    pos = sum(labels)
    neg = len(labels) - pos
    sample_weights = [neg / pos if l else 1.0 for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        RoutePhotoDataset(train_records, TRAIN_TRANSFORM),
        batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        RoutePhotoDataset(val_records, VAL_TRANSFORM),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"Train: {len(train_df)} ({sum(labels)} pos)  |  Val: {len(val_df)}")
    return train_loader, val_loader
