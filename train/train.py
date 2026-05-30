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
import argparse
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
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
    proj_dim: int = 128             # projection head output dim; set to 0 to skip the head entirely
    img_size: int = 224             # input resolution; 336 gives richer features but needs smaller batch
    num_unfrozen_blocks: int = 0    # unfreeze last N transformer blocks (0 = fully frozen backbone)

    # Training
    n_views: int = 2                # augmented views per image per batch
    batch_size: int = 128           # actual GPU batch = batch_size (sampler divides by n_views internally)
    lr: float = 2e-4                # projection head LR
    backbone_lr: float = 1e-5      # unfrozen backbone block LR — much lower to avoid destroying pretrained features
    temperature: float = 0.07
    weight_decay: float = 1e-4
    head_only_epochs: int = 5       # epochs to train only the projection head before unfreezing backbone blocks
    stage1_epochs: int = 150        # high ceiling — early stopping decides when to quit
    stage2_epochs: int = 0          # set >0 to run Stage 2 after Stage 1 saturates
    warmup_epochs: int = 3          # short warmup — no need to ramp to a large LR
    patience: int = 15
    gradient_clip: float = 1.0
    precision: int = 16

    # Eval
    recall_every_n_epochs: int = 1  # every epoch
    train_split: float = 0.8        # fraction of images to use for training
    val_split: float = 0.1          # fraction of images to use for validation
    val1_batch_size: int = 0        # Stage 1 val gallery size (0 = use batch_size)
    val2_batch_size: int = 0        # Stage 2 val gallery size (0 = use batch_size)

    # Local image directory (set when training from B2-downloaded images)
    # If set, image_dir + manifest_path are used instead of hf_dataset
    image_dir: str = ""
    manifest_path: str = ""           # CSV with image_id, route_id, area_id, b2_key, label

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
    def __init__(self, embed_dim=384, proj_dim=128, img_size=224, lr=2e-4, backbone_lr=1e-5,
                 temperature=0.07, weight_decay=1e-4, warmup_epochs=3,
                 num_unfrozen_blocks=0, backbone_name="vit_small_patch14_dinov2.lvd142m"):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.backbone_lr = backbone_lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, img_size=img_size
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        if num_unfrozen_blocks > 0:
            for block in self.backbone.blocks[-num_unfrozen_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        # proj_dim=0 → no projection head; SupConLoss runs directly on backbone output
        if proj_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, proj_dim),
            )
        else:
            self.proj = None

        self.loss_fn = SupConLoss(temperature=temperature)

    def encode(self, x):
        """L2-normalised embedding — use this for DB storage and KNN queries."""
        feat = self.backbone(x)
        if self.proj is not None:
            feat = self.proj(feat)
        return F.normalize(feat, dim=-1)

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
        imgs, labels = batch
        return {"emb": self(imgs).detach(), "labels": labels}

    def configure_optimizers(self):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        proj_params = list(self.proj.parameters()) if self.proj is not None else []
        param_groups = [{"params": proj_params, "lr": self.lr}] if proj_params else []
        if backbone_params:
            lr = self.lr if not param_groups else self.backbone_lr
            param_groups.append({"params": backbone_params, "lr": lr})
        opt = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        sched = _make_lr_scheduler(opt, self.warmup_epochs, self.trainer.max_epochs)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]


# ── Recall@K Callback ─────────────────────────────────────────────────────────

class RecallAtKCallback(pl.Callback):
    """
    Batch-level Recall@K, MRR, and alignment — computed within each val batch.

    Each batch acts as its own retrieval gallery, which matches production semantics
    (search against a few hundred climbs, not thousands). Hooks into PL's validation
    loop via on_validation_batch_end so the val loader is only iterated once.
    """
    def __init__(self, every_n_epochs=1, ks=(1, 3, 5)):
        self.every_n_epochs = every_n_epochs
        self.ks = ks
        self._batch_data: list = []  # list of (emb: Tensor, labels: list[int])

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if outputs is None:
            return
        emb = outputs["emb"].cpu()
        labels = outputs["labels"].tolist() if hasattr(outputs["labels"], "tolist") else list(outputs["labels"])
        self._batch_data.append((emb, labels))

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._batch_data:
            return

        recall_hits = {k: 0 for k in self.ks}
        mrr_sum = 0.0
        total = 0
        align_vals = []

        for emb, labels in self._batch_data:
            B = len(labels)
            if B < 2:
                continue

            # Cosine similarity — embeddings are already L2-normalised
            sim = emb @ emb.T  # (B, B)

            for i in range(B):
                sims_i = sim[i].clone()
                sims_i[i] = -float("inf")  # exclude self
                ranked = sims_i.argsort(descending=True).tolist()

                for k in self.ks:
                    if labels[i] in [labels[j] for j in ranked[:k]]:
                        recall_hits[k] += 1
                for rank, j in enumerate(ranked, start=1):
                    if labels[j] == labels[i]:
                        mrr_sum += 1.0 / rank
                        break

            # Alignment: mean sq L2 dist between same-route pairs in this batch
            route_embs: dict = defaultdict(list)
            for i, label in enumerate(labels):
                route_embs[label].append(i)
            for idxs in route_embs.values():
                if len(idxs) < 2:
                    continue
                e = emb[idxs]
                sq_dist = 2 - 2 * (e @ e.T)  # ||a-b||^2 = 2 - 2cos for unit vectors
                mask = torch.triu(torch.ones(len(idxs), len(idxs)), diagonal=1).bool()
                align_vals.append(sq_dist[mask].mean().item())

            total += B

        self._batch_data.clear()

        if total == 0:
            return

        metrics = {f"val_recall@{k}": recall_hits[k] / total for k in self.ks}
        metrics["val_mrr"] = mrr_sum / total

        for key, val in metrics.items():
            pl_module.log(key, val, prog_bar=(key == "val_recall@1"))
        trainer.callback_metrics.update({k: torch.tensor(v) for k, v in metrics.items()})

        if align_vals:
            pl_module.log("val_alignment", sum(align_vals) / len(align_vals))


# ── Training ──────────────────────────────────────────────────────────────────

def _build_loaders(cfg):
    """Returns (train_loader, train_loader_hard, val_loader_1, val_loader_2, test_loader).

    Stage 1 loaders use random batch sampling (easy negatives).
    Stage 2 loaders use proximity-based batch sampling (hard negatives).
    """
    samples_per_batch = max(1, cfg.batch_size // cfg.n_views)
    val1_bs = cfg.val1_batch_size or cfg.batch_size
    val2_bs = cfg.val2_batch_size or cfg.batch_size

    token = cfg.hf_token or os.environ.get("HF_TOKEN")
    ds = load_dataset(cfg.hf_dataset, token=token)

    train_split, val_split, test_split = create_split(ds["train"], cfg.train_split, cfg.val_split)
    print(f"Images — train: {len(train_split)}  val: {len(val_split)}  test: {len(test_split)}")

    train_loader = DataLoader(
        SupConDataset(train_split, n_views=cfg.n_views, img_size=cfg.img_size),
        batch_sampler=MultiRouteBatchSampler(train_split, samples_per_batch),
        collate_fn=supcon_collate, num_workers=cfg.num_workers, pin_memory=True,
    )
    train_loader_hard = DataLoader(
        SupConDataset(train_split, n_views=cfg.n_views, img_size=cfg.img_size),
        batch_sampler=HardNegativeBatchSampler(train_split, samples_per_batch),
        collate_fn=supcon_collate, num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader_1 = DataLoader(
        EvalDataset(val_split, img_size=cfg.img_size),
        batch_sampler=MultiRouteBatchSampler(val_split, val1_bs, shuffle=False),
        num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader_2 = DataLoader(
        EvalDataset(val_split, img_size=cfg.img_size),
        batch_sampler=HardNegativeBatchSampler(val_split, val2_bs, shuffle=False),
        num_workers=cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        EvalDataset(test_split, img_size=cfg.img_size),
        batch_sampler=HardNegativeBatchSampler(test_split, cfg.batch_size, shuffle=False),
        num_workers=cfg.num_workers,
    )

    return train_loader, train_loader_hard, val_loader_1, val_loader_2, test_loader


def train(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    train_loader, train_loader_hard, val_loader_1, val_loader_2, test_loader = _build_loaders(cfg)

    # Head warmup only makes sense when there's a projection head to train in isolation.
    # Without a head (proj_dim=0), start with backbone blocks already unfrozen.
    do_head_warmup = cfg.proj_dim > 0 and cfg.head_only_epochs > 0 and cfg.num_unfrozen_blocks > 0
    initial_unfrozen = 0 if do_head_warmup else cfg.num_unfrozen_blocks

    model = RouteFinderModel(
        embed_dim=cfg.embed_dim, proj_dim=cfg.proj_dim, img_size=cfg.img_size,
        lr=cfg.lr, backbone_lr=cfg.backbone_lr, temperature=cfg.temperature,
        weight_decay=cfg.weight_decay, warmup_epochs=cfg.warmup_epochs,
        num_unfrozen_blocks=initial_unfrozen, backbone_name=cfg.backbone,
    )

    logger = CSVLogger(cfg.checkpoint_dir, name="", version="")

    def _make_trainer(max_epochs, stage_name):
        ckpt = ModelCheckpoint(
            monitor="val_recall@1", dirpath=cfg.checkpoint_dir,
            filename=f"{stage_name}" + "-{epoch:02d}-{val_recall@1:.3f}",
            save_top_k=1, mode="max", save_last=(stage_name == "stage2"),
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs, accelerator="gpu", devices=1,
            precision=cfg.precision, log_every_n_steps=1,
            gradient_clip_val=cfg.gradient_clip,
            logger=logger,
            callbacks=[
                ckpt,
                LearningRateMonitor("epoch"),
                RecallAtKCallback(every_n_epochs=cfg.recall_every_n_epochs),
                EarlyStopping("val_recall@1", patience=cfg.patience, mode="max", strict=False),
            ],
        )
        return trainer, ckpt

    # ── Head warmup: frozen backbone, projection head only ────────────────────
    if do_head_warmup:
        trainer_head, ckpt_head = _make_trainer(cfg.head_only_epochs, "stage1_head")
        trainer_head.fit(model, train_loader, val_loader_1)
        if ckpt_head.best_model_path:
            model = RouteFinderModel.load_from_checkpoint(
                ckpt_head.best_model_path,
                num_unfrozen_blocks=cfg.num_unfrozen_blocks,
                backbone_lr=cfg.backbone_lr,
            )

    # ── Stage 1: random batches, easy negatives ───────────────────────────────
    trainer_s1, ckpt_s1 = _make_trainer(cfg.stage1_epochs, "stage1")
    trainer_s1.fit(model, train_loader, val_loader_1)

    if ckpt_s1.best_model_path:
        model = RouteFinderModel.load_from_checkpoint(ckpt_s1.best_model_path)

    metrics_csv = os.path.join(cfg.checkpoint_dir, "metrics.csv")
    print(f"\nBest Stage 1 checkpoint: {ckpt_s1.best_model_path}")

    if cfg.stage2_epochs <= 0:
        return ckpt_s1.best_model_path, metrics_csv, test_loader

    # ── Stage 2: mixed hard+easy batches ─────────────────────────────────────
    # configure_optimizers is re-called by the new Trainer, giving Stage 2 its
    # own warmup + cosine cycle — appropriate for the harder task.
    trainer_s2, ckpt_s2 = _make_trainer(cfg.stage2_epochs, "stage2")
    trainer_s2.fit(model, train_loader_hard, val_loader_2)

    print(f"Best Stage 2 checkpoint: {ckpt_s2.best_model_path}")
    return ckpt_s2.best_model_path or ckpt_s1.best_model_path, metrics_csv, test_loader


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--hf_dataset", default="DeclanBracken/RouteFinderDatasetV2")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_only_epochs", type=int, default=5)
    parser.add_argument("--stage1_epochs", type=int, default=150)
    parser.add_argument("--stage2_epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_unfrozen_blocks", type=int, default=0)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    train(Config(
        hf_token=args.hf_token,
        hf_dataset=args.hf_dataset,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        head_only_epochs=args.head_only_epochs,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        num_unfrozen_blocks=args.num_unfrozen_blocks,
        checkpoint_dir=args.checkpoint_dir,
    ))
