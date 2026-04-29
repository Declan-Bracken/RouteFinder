"""
Route retrieval training — Kaggle-portable two-phase pipeline.

Phase 1 — SimCLR pretraining on RouteFinderDataset (~8k noisy MP images)
Phase 2 — SupCon fine-tuning on RouteFinderDatasetV2 (~573 curated multiview pairs)

On Kaggle:
  Add HF token to Kaggle secrets as HF_TOKEN.
  Run: python train.py --phase both
  Or skip phase 1: python train.py --phase supcon --simclr_ckpt /path/to/simclr.ckpt
"""

import os
import random
import argparse
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.neighbors import NearestNeighbors
from datasets import load_dataset
import timm
from pytorch_metric_learning.losses import SupConLoss
from lightly.loss import NTXentLoss


# ── Configs ───────────────────────────────────────────────────────────────────

@dataclass
class SimCLRConfig:
    hf_dataset: str = "DeclanBracken/RouteFinderDataset"
    batch_size: int = 64
    lr: float = 5e-4
    temperature: float = 0.1
    weight_decay: float = 1e-4
    max_epochs: int = 50
    warmup_epochs: int = 5
    patience: int = 10
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"
    precision: int = 16


@dataclass
class SupConConfig:
    hf_dataset: str = "DeclanBracken/RouteFinderDatasetV2"
    simclr_ckpt: str = None         # path to SimCLR checkpoint; if None, use ImageNet weights
    batch_size: int = 256
    lr: float = 1e-4
    temperature: float = 0.07
    weight_decay: float = 2e-4
    num_unfrozen_blocks: int = 1    # how many of resnet's 4 layer groups to unfreeze
    n_views: int = 2                # augmented views per image — key multiplier for small datasets
    max_epochs: int = 100
    warmup_epochs: int = 5
    patience: int = 15
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"
    precision: int = 16
    gradient_clip: float = 1.0
    recall_every_n_epochs: int = 5  # how often to compute Recall@K on val set
    test_split: float = 0.2


# ── Augmentations ─────────────────────────────────────────────────────────────
#
# Key domain gap issues for field photos:
#   - Camera held at any angle          → RandomPerspective + RandomRotation
#   - Variable distance from wall       → RandomResizedCrop (wide scale range)
#   - Outdoor lighting variance         → ColorJitter + RandomGrayscale
#   - Blur from hand movement           → GaussianBlur

SIMCLR_TRANSFORM = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224, scale=(0.4, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    T.RandomPerspective(distortion_scale=0.3, p=0.5),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SUPCON_TRANSFORM = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    T.RandomPerspective(distortion_scale=0.2, p=0.3),
    T.RandomRotation(10),
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

class SimCLRDataset(Dataset):
    """Returns two independently augmented views of each image."""
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["image"].convert("RGB")
        return SIMCLR_TRANSFORM(img), SIMCLR_TRANSFORM(img)


class SupConDataset(Dataset):
    """
    Returns n_views augmented views per image as a stacked tensor (n_views, C, H, W).
    Multiple views of the same image act as additional positives in SupCon,
    which is critical when fine-tuning data is scarce.
    """
    def __init__(self, hf_dataset, n_views=2):
        self.ds = hf_dataset
        self.n_views = n_views

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"].convert("RGB")
        views = torch.stack([SUPCON_TRANSFORM(img) for _ in range(self.n_views)])
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
    """Return list-of-lists: each inner list is indices of one route."""
    groups = defaultdict(list)
    for i, sample in enumerate(hf_dataset):
        groups[sample["label"]].append(i)
    return list(groups.values())


class MultiRouteBatchSampler(BatchSampler):
    """
    Packs multiple full routes into each batch.
    SupCon requires multiple samples per class within a batch to form positives.
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


def supcon_collate(batch):
    views, labels = zip(*batch)
    return torch.stack(views), torch.stack(labels)


# ── Models ────────────────────────────────────────────────────────────────────

def _make_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


class SimCLREncoder(pl.LightningModule):
    def __init__(self, lr=5e-4, temperature=0.1, weight_decay=1e-4, warmup_epochs=5):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.proj = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(inplace=True), nn.Linear(2048, 128)
        )
        self.criterion = NTXentLoss(temperature=temperature)

    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), dim=-1)

    def training_step(self, batch, _):
        x1, x2 = batch
        loss = self.criterion(self(x1), self(x2))
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x1, x2 = batch
        loss = self.criterion(self(x1), self(x2))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = _make_warmup_cosine_scheduler(opt, self.hparams.warmup_epochs, self.trainer.max_epochs)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]


class SupConModel(pl.LightningModule):
    def __init__(self, backbone, lr=1e-4, temperature=0.07, weight_decay=2e-4,
                 num_unfrozen_blocks=1, warmup_epochs=5):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.loss_fn = SupConLoss(temperature=temperature)

        for p in self.backbone.parameters():
            p.requires_grad = False

        layers = [self.backbone.layer1, self.backbone.layer2,
                  self.backbone.layer3, self.backbone.layer4]
        for layer in layers[-num_unfrozen_blocks:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Keep BatchNorm stats fixed in frozen layers
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        return F.normalize(self.backbone(x), dim=-1)

    def training_step(self, batch, _):
        views, labels = batch           # views: (B, n_views, C, H, W)
        B, V, C, H, W = views.shape
        z = self(views.view(B * V, C, H, W))
        labels_expanded = labels.unsqueeze(1).expand(B, V).reshape(B * V).to(self.device)
        loss = self.loss_fn(z, labels_expanded)
        self.log("supcon_train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        views, labels = batch
        B, V, C, H, W = views.shape
        z = self(views.view(B * V, C, H, W))
        labels_expanded = labels.unsqueeze(1).expand(B, V).reshape(B * V).to(self.device)
        loss = self.loss_fn(z, labels_expanded)
        self.log("supcon_val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
        )
        sched = _make_warmup_cosine_scheduler(opt, self.hparams.warmup_epochs, self.trainer.max_epochs)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]


# ── Recall@K Callback ─────────────────────────────────────────────────────────

class RecallAtKCallback(pl.Callback):
    """
    Computes Recall@K on the val split every N epochs using the backbone directly.
    This is the metric that actually matters for retrieval — not contrastive loss.
    """
    def __init__(self, val_hf_dataset, every_n_epochs=5, ks=(1, 3, 5)):
        self.val_ds = val_hf_dataset
        self.every_n_epochs = every_n_epochs
        self.ks = ks

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        loader = DataLoader(EvalDataset(self.val_ds), batch_size=64, num_workers=2)
        device = next(pl_module.parameters()).device
        pl_module.eval()

        embeddings, labels = [], []
        for imgs, lbls in loader:
            z = pl_module(imgs.to(device))
            embeddings.append(z.cpu())
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
            print(f"  Recall@{k}: {recall:.4f}")


# ── Training phases ───────────────────────────────────────────────────────────

def train_simclr(cfg: SimCLRConfig, hf_token: str = None):
    token = hf_token or os.environ.get("HF_TOKEN")
    ds = load_dataset(cfg.hf_dataset, token=token)
    train_loader = DataLoader(
        SimCLRDataset(ds["train"]), batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        SimCLRDataset(ds["val"]), batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
    )
    model = SimCLREncoder(
        lr=cfg.lr, temperature=cfg.temperature,
        weight_decay=cfg.weight_decay, warmup_epochs=cfg.warmup_epochs,
    )
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", dirpath=cfg.checkpoint_dir,
        filename="simclr-{epoch:02d}-{val_loss:.3f}",
        save_top_k=1, mode="min", save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs, accelerator="gpu", devices=1,
        precision=cfg.precision, log_every_n_steps=1,
        callbacks=[
            ckpt_cb,
            EarlyStopping("val_loss", patience=cfg.patience, mode="min"),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"Best SimCLR checkpoint: {ckpt_cb.best_model_path}")
    return ckpt_cb.best_model_path


def train_supcon(cfg: SupConConfig, simclr_ckpt: str = None, hf_token: str = None):
    token = hf_token or os.environ.get("HF_TOKEN")
    ds = load_dataset(cfg.hf_dataset, token=token)

    all_routes = list(set(ds["train"]["label"]))
    random.seed(42)
    test_routes = set(random.sample(all_routes, max(1, int(len(all_routes) * cfg.test_split))))
    train_split = ds["train"].filter(lambda x: x["label"] not in test_routes)
    test_split  = ds["train"].filter(lambda x: x["label"] in test_routes)

    print(f"SupCon train: {len(train_split)} images | val: {len(test_split)} images")

    train_loader = DataLoader(
        SupConDataset(train_split, n_views=cfg.n_views),
        batch_sampler=MultiRouteBatchSampler(group_by_label(train_split), cfg.batch_size),
        collate_fn=supcon_collate, num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        SupConDataset(test_split, n_views=cfg.n_views),
        batch_sampler=MultiRouteBatchSampler(group_by_label(test_split), cfg.batch_size, shuffle=False),
        collate_fn=supcon_collate, num_workers=cfg.num_workers, pin_memory=True,
    )

    ckpt_path = simclr_ckpt or cfg.simclr_ckpt
    if ckpt_path:
        backbone = SimCLREncoder.load_from_checkpoint(ckpt_path).backbone
        print(f"Loaded SimCLR backbone from {ckpt_path}")
    else:
        backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        print("No SimCLR checkpoint — using ImageNet-pretrained backbone")

    model = SupConModel(
        backbone=backbone, lr=cfg.lr, temperature=cfg.temperature,
        weight_decay=cfg.weight_decay, num_unfrozen_blocks=cfg.num_unfrozen_blocks,
        warmup_epochs=cfg.warmup_epochs,
    )
    ckpt_cb = ModelCheckpoint(
        monitor="supcon_val_loss", dirpath=cfg.checkpoint_dir,
        filename="supcon-{epoch:02d}-{supcon_val_loss:.3f}",
        save_top_k=1, mode="min", save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs, accelerator="gpu", devices=1,
        precision=cfg.precision, log_every_n_steps=1,
        gradient_clip_val=cfg.gradient_clip,
        callbacks=[
            ckpt_cb,
            EarlyStopping("supcon_val_loss", patience=cfg.patience, mode="min"),
            LearningRateMonitor("epoch"),
            RecallAtKCallback(test_split, every_n_epochs=cfg.recall_every_n_epochs),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"Best SupCon checkpoint: {ckpt_cb.best_model_path}")
    return ckpt_cb.best_model_path


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["simclr", "supcon", "both"], default="both",
                        help="Which phase(s) to run")
    parser.add_argument("--simclr_ckpt", default=None,
                        help="Existing SimCLR checkpoint — skips phase 1 if provided with --phase supcon")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    simclr_ckpt = args.simclr_ckpt
    if args.phase in ("simclr", "both"):
        simclr_ckpt = train_simclr(SimCLRConfig(), hf_token=args.hf_token)
    if args.phase in ("supcon", "both"):
        train_supcon(SupConConfig(), simclr_ckpt=simclr_ckpt, hf_token=args.hf_token)
