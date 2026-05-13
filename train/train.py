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

from train.samplers import create_split, MultiRouteBatchSampler, HardNegativeBatchSampler
from train.datasets import supcon_collate, SupConDataset, EvalDataset

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Data
    hf_dataset: str = "DeclanBracken/RouteFinderDatasetV3"
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
    recall_every_n_epochs: int = 1  # every epoch
    train_split: float = 0.8        # fraction of images to use for training
    val_split: float = 0.1          # fraction of images to use for validation

    # Local image directory (set when training from B2-downloaded images)
    # If set, image_dir + manifest_path are used instead of hf_dataset
    image_dir: str = ""
    manifest_path: str = ""           # CSV with image_id, route_id, area_id, b2_key, label
    use_area_sampler: bool = True     # area-aware hard negative sampling

    # I/O
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"


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
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.weight_decay = weight_decay
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
        pass

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        sched = _make_lr_scheduler(opt, self.warmup_epochs, self.trainer.max_epochs)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]


# ── Recall@K Callback ─────────────────────────────────────────────────────────

class RecallAtKCallback(pl.Callback):
    """
    Computes Recall@K on the held-out val split every N epochs.
    """
    def __init__(self, loader, every_n_epochs=5, ks=(1, 3, 5)):
        self.every_n_epochs = every_n_epochs
        self.ks = ks
        self.loader = loader

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        device = next(pl_module.parameters()).device
        pl_module.eval()

        embeddings, labels = [], []
        for imgs, lbls in self.loader:
            embeddings.append(pl_module(imgs.to(device)).cpu())
            labels.extend(lbls.tolist() if hasattr(lbls, "tolist") else lbls)

        emb = torch.cat(embeddings)
        knn = NearestNeighbors(n_neighbors=max(self.ks) + 1, metric="cosine")
        knn.fit(emb.numpy())
        _, indices = knn.kneighbors(emb.numpy())

        # Recall@K and MRR in one pass
        max_k = max(self.ks)
        recall_hits = {k: 0 for k in self.ks}
        mrr = 0.0
        for i in range(len(labels)):
            neighbors = [labels[j] for j in indices[i][1:max_k + 1]]
            for k in self.ks:
                if labels[i] in neighbors[:k]:
                    recall_hits[k] += 1
            for rank, label in enumerate(neighbors, start=1):
                if label == labels[i]:
                    mrr += 1.0 / rank
                    break

        for k in self.ks:
            pl_module.log(f"val_recall@{k}", recall_hits[k] / len(labels), prog_bar=(k == 1))
        pl_module.log("val_mrr", mrr / len(labels))

        # Alignment: mean squared L2 distance between same-route pairs
        # Lower = tighter positive clusters
        route_embs = defaultdict(list)
        for i, label in enumerate(labels):
            route_embs[label].append(i)

        align_vals = []
        for idxs in route_embs.values():
            if len(idxs) < 2:
                continue
            e = emb[idxs]
            sq_dist = 2 - 2 * (e @ e.T)  # ||a-b||^2 = 2 - 2cos for unit vectors
            mask = torch.triu(torch.ones(len(idxs), len(idxs)), diagonal=1).bool()
            align_vals.append(sq_dist[mask].mean().item())
        pl_module.log("val_alignment", sum(align_vals) / len(align_vals) if align_vals else 0.0)

        # Uniformity: log mean exp(-2||a-b||^2) over random pairs
        # More negative = more uniform spread across hypersphere
        sample = emb if len(emb) <= 2000 else emb[torch.randperm(len(emb))[:2000]]
        sq_dists = torch.cdist(sample, sample).pow(2)
        mask = torch.triu(torch.ones(len(sample), len(sample)), diagonal=1).bool()
        pl_module.log("val_uniformity", sq_dists[mask].mul(-2).exp().mean().log().item())


# ── Training ──────────────────────────────────────────────────────────────────

 
def _build_loaders(cfg):

    """Returns (train_loader, val_loader, val_eval_ds)."""
    samples_per_batch = max(1, cfg.batch_size // cfg.n_views)

    token = cfg.hf_token or os.environ.get("HF_TOKEN")
    ds = load_dataset(cfg.hf_dataset, token=token)

    train_split, val_split, test_split = create_split(ds["train"], cfg.train_split, cfg.val_split)
    print(f"Images — train: {len(train_split)}  val: {len(val_split)} test: {len(test_split)}")
    
    # Random cross-area route sampling
    train_loader = DataLoader(
        SupConDataset(train_split, n_views=cfg.n_views),
        batch_sampler=MultiRouteBatchSampler(train_split, samples_per_batch),
        collate_fn=supcon_collate, num_workers=cfg.num_workers, pin_memory=True,
    )

    # Same construction, but with hard negative sampler
    train_loader_hard = DataLoader(
        SupConDataset(train_split, n_views=cfg.n_views),
        batch_sampler=HardNegativeBatchSampler(train_split, samples_per_batch),
        collate_fn=supcon_collate, num_workers=cfg.num_workers, pin_memory=True,
    )

    val_loader = DataLoader(
        EvalDataset(val_split),
        batch_sampler=HardNegativeBatchSampler(
            val_split, cfg.batch_size, shuffle=False
        ),
        num_workers=cfg.num_workers, pin_memory=True,
    )

    test_loader = DataLoader(
        EvalDataset(test_split),
        batch_sampler=HardNegativeBatchSampler(
            test_split, cfg.batch_size, shuffle=False
        ),
    )

    return train_loader, train_loader_hard, val_loader, test_loader


def train(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    train_loader, train_loader_hard, val_loader, test_loader = _build_loaders(cfg)

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
            RecallAtKCallback(val_loader, every_n_epochs=cfg.recall_every_n_epochs),
            EarlyStopping("val_recall@1", patience=cfg.patience, mode="max", strict=False),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    metrics_csv = os.path.join(cfg.checkpoint_dir, "metrics.csv")
    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")
    return ckpt_cb.best_model_path, metrics_csv, test_loader


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
