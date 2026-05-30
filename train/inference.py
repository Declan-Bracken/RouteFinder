"""
Inference-only module — imported by the API server and the Kaggle eval cell.
Matches RouteFinderModel in train.py exactly so checkpoints load without errors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pytorch_lightning as pl

from train.datasets import get_eval_transform

EVAL_TRANSFORM = get_eval_transform(224)  # default; override for non-224 checkpoints


class RouteFinderModel(pl.LightningModule):
    def __init__(self, embed_dim=384, proj_dim=128, img_size=224, lr=2e-4, backbone_lr=1e-5,
                 temperature=0.07, weight_decay=1e-4, warmup_epochs=3,
                 num_unfrozen_blocks=0, backbone_name="vit_small_patch14_dinov2.lvd142m"):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, img_size=img_size
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        if num_unfrozen_blocks > 0:
            for block in self.backbone.blocks[-num_unfrozen_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        if proj_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, proj_dim),
            )
        else:
            self.proj = None

    def encode(self, x):
        feat = self.backbone(x)
        if self.proj is not None:
            feat = self.proj(feat)
        return F.normalize(feat, dim=-1)

    def forward(self, x):
        return self.encode(x)
