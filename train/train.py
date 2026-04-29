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
from dataclasses import dataclass
from collections import defaultdict

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
    recall_every_n_epochs: int = 5  # how often Recall@K is computed on the val split
    test_split: float = 0.2         # fraction of routes held out for validation

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

def train(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    token = cfg.hf_token or os.environ.get("HF_TOKEN")
    ds = load_dataset(cfg.hf_dataset, token=token)

    all_routes = list(set(ds["train"]["label"]))
    random.seed(42)
    n_test = max(1, int(len(all_routes) * cfg.test_split))
    test_routes = set(random.sample(all_routes, n_test))
    train_split = ds["train"].filter(lambda x: x["label"] not in test_routes)
    test_split  = ds["train"].filter(lambda x: x["label"] in test_routes)

    print(f"Routes — train: {len(all_routes) - n_test}  val: {n_test}")
    print(f"Images — train: {len(train_split)}  val: {len(test_split)}")

    # Sampler divides by n_views because each sample expands to n_views GPU images
    samples_per_batch = max(1, cfg.batch_size // cfg.n_views)
    train_loader = DataLoader(
        SupConDataset(train_split, n_views=cfg.n_views),
        batch_sampler=MultiRouteBatchSampler(group_by_label(train_split), samples_per_batch),
        collate_fn=supcon_collate, num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        SupConDataset(test_split, n_views=cfg.n_views),
        batch_sampler=MultiRouteBatchSampler(group_by_label(test_split), samples_per_batch, shuffle=False),
        collate_fn=supcon_collate, num_workers=cfg.num_workers, pin_memory=True,
    )

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
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs, accelerator="gpu", devices=1,
        precision=cfg.precision, log_every_n_steps=1,
        gradient_clip_val=cfg.gradient_clip,
        callbacks=[
            ckpt_cb,
            EarlyStopping("val_recall@1", patience=cfg.patience, mode="max"),
            LearningRateMonitor("epoch"),
            RecallAtKCallback(test_split, every_n_epochs=cfg.recall_every_n_epochs),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")
    return ckpt_cb.best_model_path


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
