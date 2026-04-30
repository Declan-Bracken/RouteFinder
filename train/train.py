"""
Route retrieval training — DINOv2-Small backbone + SupCon projection head.

Architecture:
  DINOv2-S/14 (frozen, 384-d) → projection head (384 → 128, L2-norm) → SupConLoss

Why this instead of training SimCLR from scratch:
  DINOv2 was pre-trained on 142M images with a far richer objective. Fine-tuning
  only a small projection head with SupCon gives better retrieval than a SimCLR
  backbone trained on ~8k noisy images, and trains in minutes instead of hours.

On Kaggle:
  1. Enable internet access in notebook settings
  2. Add HF_TOKEN to Kaggle secrets (Add-ons → Secrets)
  3. Run the kaggle_train.ipynb notebook — it handles setup and calls this script
"""

import os
import random
import argparse
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler
from PIL import Image
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from sklearn.neighbors import NearestNeighbors
from datasets import load_dataset
import timm
from pytorch_metric_learning.losses import SupConLoss


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Data
    hf_dataset: str = "DeclanBracken/RouteFinderDatasetV2"
    hf_token: str = None            # or set HF_TOKEN env var

    # Model
    backbone: str = "vit_small_patch14_dinov2.lvd142m"
    embed_dim: int = 384            # DINOv2-S output dim — don't change unless switching backbone
    proj_dim: int = 128             # projection head output dim
    num_unfrozen_blocks: int = 0    # unfreeze last N transformer blocks (0 = fully frozen backbone)

    # Training
    n_views: int = 2                # augmented views per image per batch
    batch_size: int = 128           # actual GPU batch = batch_size (sampler divides by n_views internally)
    lr: float = 1e-3                # higher LR is fine — only the small proj head is being trained
    temperature: float = 0.07
    weight_decay: float = 1e-4
    max_epochs: int = 100
    warmup_epochs: int = 5
    patience: int = 15
    gradient_clip: float = 1.0
    precision: int = 16

    # Eval
    recall_every_n_epochs: int = 1  # every epoch — val set is small so overhead is negligible
    test_split: float = 0.2         # fraction of routes held out for validation

    # Local image directory (set when training from B2-downloaded images)
    # If set, image_dir + manifest_path are used instead of hf_dataset
    image_dir: str = ""
    manifest_path: str = ""           # CSV with image_id, route_id, area_id, b2_key, label
    use_area_sampler: bool = True     # area-aware hard negative sampling

    # I/O
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"


# ── Augmentations ─────────────────────────────────────────────────────────────
#
# Key domain gap between MP photos and real field captures:
#   - Phone held at any angle          → RandomPerspective + RandomRotation
#   - Variable distance from wall      → RandomResizedCrop (wide scale range)
#   - Outdoor lighting changes         → ColorJitter + RandomGrayscale
#   - Hand movement / focus blur       → GaussianBlur

TRAIN_TRANSFORM = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.RandomGrayscale(p=0.1),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    T.RandomPerspective(distortion_scale=0.3, p=0.5),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Datasets ──────────────────────────────────────────────────────────────────

class SupConDataset(Dataset):
    """
    Returns n_views independently augmented views per image, stacked as (n_views, C, H, W).
    With only ~573 fine-tuning images, multiple views per image are the primary way
    to increase the number of positive pairs the loss sees per epoch.
    """
    def __init__(self, hf_dataset, n_views=2):
        self.ds = hf_dataset
        self.n_views = n_views

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"].convert("RGB")
        views = torch.stack([TRAIN_TRANSFORM(img) for _ in range(self.n_views)])
        return views, torch.tensor(sample["label"], dtype=torch.long)


class EvalDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        return EVAL_TRANSFORM(sample["image"].convert("RGB")), sample["label"]


def group_by_label(hf_dataset):
    """Returns list-of-lists: each inner list holds the indices of one route."""
    groups = defaultdict(list)
    for i, sample in enumerate(hf_dataset):
        groups[sample["label"]].append(i)
    return list(groups.values())


class MultiRouteBatchSampler(BatchSampler):
    """
    Packs several complete routes into each batch. SupCon requires multiple
    samples per class within a batch — without this, most anchors have no
    positives and the loss is meaningless.
    """
    def __init__(self, route_groups, max_batch_size, shuffle=True):
        self.groups = route_groups
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle

    def __iter__(self):
        groups = list(self.groups)
        if self.shuffle:
            random.shuffle(groups)
        batch, batch_len = [], 0
        for group in groups:
            if batch_len + len(group) > self.max_batch_size and batch:
                yield batch
                batch, batch_len = [], 0
            batch.extend(group)
            batch_len += len(group)
        if batch:
            yield batch

    def __len__(self):
        total = sum(len(g) for g in self.groups)
        return max(1, (total + self.max_batch_size - 1) // self.max_batch_size)


class AreaAwareBatchSampler(BatchSampler):
    """
    Hard negative sampler: packs routes from the same area into each batch.
    Routes from the same crag share rock type, color, and vegetation — forcing
    the model to learn fine-grained discriminative features rather than easy
    cross-crag differences.
    """
    def __init__(self, area_route_groups: dict, max_batch_size: int, shuffle: bool = True):
        # area_route_groups: {area_id: [[indices_route1], [indices_route2], ...]}
        self.area_route_groups = area_route_groups
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle

    def __iter__(self):
        areas = list(self.area_route_groups.values())
        if self.shuffle:
            random.shuffle(areas)
        for route_groups in areas:
            groups = list(route_groups)
            if self.shuffle:
                random.shuffle(groups)
            batch, batch_len = [], 0
            for group in groups:
                if batch_len + len(group) > self.max_batch_size and batch:
                    yield batch
                    batch, batch_len = [], 0
                batch.extend(group)
                batch_len += len(group)
            if batch:
                yield batch

    def __len__(self):
        total = sum(sum(len(g) for g in groups) for groups in self.area_route_groups.values())
        return max(1, (total + self.max_batch_size - 1) // self.max_batch_size)


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


def group_by_area_and_route(manifest: pd.DataFrame) -> dict:
    """Returns {area_id: [[indices_route1], [indices_route2], ...]} for AreaAwareBatchSampler."""
    area_route = defaultdict(lambda: defaultdict(list))
    for idx, row in manifest.iterrows():
        area_route[row["area_id"]][row["route_id"]].append(idx)
    return {
        area_id: list(route_dict.values())
        for area_id, route_dict in area_route.items()
    }


def supcon_collate(batch):
    views, labels = zip(*batch)
    return torch.stack(views), torch.stack(labels)


# ── Model ─────────────────────────────────────────────────────────────────────

def _make_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup then cosine decay. Warmup prevents the chaotic early
    training instability seen when fine-tuning with a flat LR."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


class RouteFinderModel(pl.LightningModule):
    """
    Frozen DINOv2-Small backbone + small projection head trained with SupConLoss.

    At inference use model.encode(image_tensor) — the L2-normalised projection
    head output is the embedding you store in the DB and query against.

    If you later want to unfreeze and fine-tune backbone blocks, increase
    num_unfrozen_blocks (0 = fully frozen, 4 = last 4 of 12 ViT blocks unfrozen).
    With more field data, unfreezing 1-2 blocks is worth trying.
    """
    def __init__(self, embed_dim=384, proj_dim=128, lr=1e-3, temperature=0.07,
                 weight_decay=1e-4, warmup_epochs=5, num_unfrozen_blocks=0,
                 backbone_name="vit_small_patch14_dinov2.lvd142m"):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, img_size=224
        )

        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Optionally unfreeze the last N transformer blocks
        if num_unfrozen_blocks > 0:
            for block in self.backbone.blocks[-num_unfrozen_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            # Always unfreeze the final LayerNorm
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )
        self.loss_fn = SupConLoss(temperature=temperature)

    def encode(self, x):
        """L2-normalised embedding — use this for DB storage and KNN queries."""
        return F.normalize(self.proj(self.backbone(x)), dim=-1)

    def forward(self, x):
        return self.encode(x)

    def _shared_step(self, batch):
        views, labels = batch           # views: (B, n_views, C, H, W)
        B, V, C, H, W = views.shape
        z = self(views.view(B * V, C, H, W))
        labels_exp = labels.unsqueeze(1).expand(B, V).reshape(B * V).to(self.device)
        return self.loss_fn(z, labels_exp)

    def training_step(self, batch, _):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = _make_lr_scheduler(opt, self.hparams.warmup_epochs, self.trainer.max_epochs)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]


# ── Recall@K Callback ─────────────────────────────────────────────────────────

class RecallAtKCallback(pl.Callback):
    """
    Computes Recall@K on the held-out val split every N epochs.
    This is the metric that actually matters — not contrastive loss,
    which is an indirect proxy and hard to interpret across runs.
    """
    def __init__(self, val_hf_dataset, every_n_epochs=5, ks=(1, 3, 5)):
        self.val_ds = val_hf_dataset
        self.every_n_epochs = every_n_epochs
        self.ks = ks

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        # num_workers=0 avoids spawning subprocesses that re-trigger this callback
        loader = DataLoader(EvalDataset(self.val_ds), batch_size=64, num_workers=0)
        device = next(pl_module.parameters()).device
        pl_module.eval()

        embeddings, labels = [], []
        for imgs, lbls in loader:
            embeddings.append(pl_module(imgs.to(device)).cpu())
            labels.extend(lbls.tolist() if hasattr(lbls, "tolist") else lbls)

        emb = torch.cat(embeddings).numpy()
        knn = NearestNeighbors(n_neighbors=max(self.ks) + 1, metric="cosine")
        knn.fit(emb)
        _, indices = knn.kneighbors(emb)

        for k in self.ks:
            hits = sum(
                labels[i] in [labels[j] for j in indices[i][1:k + 1]]
                for i in range(len(labels))
            )
            recall = hits / len(labels)
            pl_module.log(f"val_recall@{k}", recall, prog_bar=(k == 1))


# ── Training ──────────────────────────────────────────────────────────────────

def _group_by_label_df(df: pd.DataFrame) -> list:
    """group_by_label equivalent for manifest DataFrames."""
    groups = defaultdict(list)
    for i, row in df.iterrows():
        groups[int(row["label"])].append(i)
    return list(groups.values())


def _build_loaders(cfg: Config):
    """Returns (train_loader, val_loader, val_eval_ds)."""
    samples_per_batch = max(1, cfg.batch_size // cfg.n_views)

    if cfg.image_dir and cfg.manifest_path:
        manifest = pd.read_csv(cfg.manifest_path)
        all_routes = manifest["route_id"].unique().tolist()
        random.seed(42)
        n_test = max(1, int(len(all_routes) * cfg.test_split))
        test_routes = set(random.sample(all_routes, n_test))

        train_df = manifest[~manifest["route_id"].isin(test_routes)].reset_index(drop=True)
        val_df   = manifest[manifest["route_id"].isin(test_routes)].reset_index(drop=True)

        print(f"Routes — train: {len(all_routes) - n_test}  val: {n_test}")
        print(f"Images — train: {len(train_df)}  val: {len(val_df)}")

        train_sampler = (
            AreaAwareBatchSampler(group_by_area_and_route(train_df), samples_per_batch)
            if cfg.use_area_sampler
            else MultiRouteBatchSampler(_group_by_label_df(train_df), samples_per_batch)
        )
        train_loader = DataLoader(
            LocalImageDataset(train_df, cfg.image_dir, n_views=cfg.n_views),
            batch_sampler=train_sampler,
            collate_fn=supcon_collate, num_workers=cfg.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            LocalImageDataset(val_df, cfg.image_dir, n_views=cfg.n_views),
            batch_sampler=MultiRouteBatchSampler(
                _group_by_label_df(val_df), samples_per_batch, shuffle=False
            ),
            collate_fn=supcon_collate, num_workers=cfg.num_workers, pin_memory=True,
        )
        val_eval_ds = LocalEvalDataset(val_df, cfg.image_dir)

    else:
        token = cfg.hf_token or os.environ.get("HF_TOKEN")
        ds = load_dataset(cfg.hf_dataset, token=token)

        all_routes = list(set(ds["train"]["label"]))
        random.seed(42)
        n_test = max(1, int(len(all_routes) * cfg.test_split))
        test_routes = set(random.sample(all_routes, n_test))
        train_split = ds["train"].filter(lambda x: x["label"] not in test_routes)
        val_split   = ds["train"].filter(lambda x: x["label"] in test_routes)

        print(f"Routes — train: {len(all_routes) - n_test}  val: {n_test}")
        print(f"Images — train: {len(train_split)}  val: {len(val_split)}")

        train_loader = DataLoader(
            SupConDataset(train_split, n_views=cfg.n_views),
            batch_sampler=MultiRouteBatchSampler(group_by_label(train_split), samples_per_batch),
            collate_fn=supcon_collate, num_workers=cfg.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            SupConDataset(val_split, n_views=cfg.n_views),
            batch_sampler=MultiRouteBatchSampler(
                group_by_label(val_split), samples_per_batch, shuffle=False
            ),
            collate_fn=supcon_collate, num_workers=cfg.num_workers, pin_memory=True,
        )
        val_eval_ds = EvalDataset(val_split)

    return train_loader, val_loader, val_eval_ds


def train(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    train_loader, val_loader, val_eval_ds = _build_loaders(cfg)

    model = RouteFinderModel(
        embed_dim=cfg.embed_dim, proj_dim=cfg.proj_dim, lr=cfg.lr,
        temperature=cfg.temperature, weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs, num_unfrozen_blocks=cfg.num_unfrozen_blocks,
        backbone_name=cfg.backbone,
    )

    ckpt_cb = ModelCheckpoint(
        monitor="val_recall@1", dirpath=cfg.checkpoint_dir,
        filename="routefinder-{epoch:02d}-{val_recall@1:.3f}",
        save_top_k=1, mode="max", save_last=True,
    )
    logger = CSVLogger(cfg.checkpoint_dir, name="", version="")
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs, accelerator="gpu", devices=1,
        precision=cfg.precision, log_every_n_steps=1,
        gradient_clip_val=cfg.gradient_clip,
        logger=logger,
        callbacks=[
            ckpt_cb,
            LearningRateMonitor("epoch"),
            RecallAtKCallback(val_eval_ds, every_n_epochs=cfg.recall_every_n_epochs),
            EarlyStopping("val_recall@1", patience=cfg.patience, mode="max", strict=False),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    metrics_csv = os.path.join(cfg.checkpoint_dir, "metrics.csv")
    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")
    return ckpt_cb.best_model_path, metrics_csv


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--hf_dataset", default="DeclanBracken/RouteFinderDatasetV2")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_unfrozen_blocks", type=int, default=0)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    train(Config(
        hf_token=args.hf_token,
        hf_dataset=args.hf_dataset,
        lr=args.lr,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        num_unfrozen_blocks=args.num_unfrozen_blocks,
        checkpoint_dir=args.checkpoint_dir,
    ))
