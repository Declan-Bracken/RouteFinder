import io
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import pandas as pd
import os
# ── Augmentations ─────────────────────────────────────────────────────────────

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_train_transform(img_size: int = 224) -> T.Compose:
    resize = int(img_size * 256 / 224)
    return T.Compose([
        T.Resize(resize),
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.RandomPerspective(distortion_scale=0.3, p=0.5),
        T.RandomRotation(15),
        T.ToTensor(),
        _NORM,
    ])


def get_eval_transform(img_size: int = 224) -> T.Compose:
    resize = int(img_size * 256 / 224)
    return T.Compose([
        T.Resize(resize),
        T.CenterCrop(img_size),
        T.ToTensor(),
        _NORM,
    ])


# Defaults (224px) kept for inference code that imports these directly
TRAIN_TRANSFORM = get_train_transform(224)
EVAL_TRANSFORM = get_eval_transform(224)


def _to_pil(img_field) -> Image.Image:
    """HF datasets may return image column as raw bytes dict or a PIL Image."""
    if isinstance(img_field, dict):
        return Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
    return img_field.convert("RGB")


# ── Datasets ──────────────────────────────────────────────────────────────────

class SupConDataset(Dataset):
    """
    Returns n_views independently augmented views per image, stacked as (n_views, C, H, W).
    """
    def __init__(self, hf_dataset, n_views=2, img_size=224):
        self.ds = hf_dataset
        self.n_views = n_views
        self.transform = get_train_transform(img_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = _to_pil(sample["image"])
        views = torch.stack([self.transform(img) for _ in range(self.n_views)])
        return views, torch.tensor(sample["route_id"], dtype=torch.long)


class EvalDataset(Dataset):
    def __init__(self, hf_dataset, img_size=224):
        self.ds = hf_dataset
        self.transform = get_eval_transform(img_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        return self.transform(_to_pil(sample["image"])), sample["route_id"]


# ── B2 Datasets ────────────────────────────────────────────────────────────────────

class LocalImageDataset(Dataset):
    """Dataset that reads images from a local directory using a manifest DataFrame."""
    def __init__(self, manifest: pd.DataFrame, image_dir: str, n_views: int = 2):
        self.manifest = manifest.reset_index(drop=True)
        self.image_dir = image_dir
        self.n_views = n_views

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, f"{row['image_id']}.jpg")).convert("RGB")
        views = torch.stack([TRAIN_TRANSFORM(img) for _ in range(self.n_views)])
        return views, torch.tensor(int(row["label"]), dtype=torch.long)


class LocalEvalDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_dir: str):
        self.manifest = manifest.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, f"{row['image_id']}.jpg")).convert("RGB")
        return EVAL_TRANSFORM(img), int(row["label"])
    
def supcon_collate(batch):
    views, labels = zip(*batch)
    return torch.stack(views), torch.stack(labels)

