"""
Lightweight inference-only module — no training deps (datasets, sklearn, pandas).
Imported by the API server; train.py is only used on Kaggle.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import pytorch_lightning as pl
from pytorch_metric_learning.losses import SupConLoss

EVAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class RouteFinderModel(pl.LightningModule):
    def __init__(self, embed_dim=384, proj_dim=128, lr=1e-3, temperature=0.07,
                 weight_decay=1e-4, warmup_epochs=5, num_unfrozen_blocks=0,
                 backbone_name="vit_small_patch14_dinov2.lvd142m"):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, img_size=224
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        if num_unfrozen_blocks > 0:
            for block in self.backbone.blocks[-num_unfrozen_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )
        self.loss_fn = SupConLoss(temperature=temperature)

    def encode(self, x):
        return F.normalize(self.proj(self.backbone(x)), dim=-1)

    def forward(self, x):
        return self.encode(x)

    def _shared_step(self, batch):
        views, labels = batch
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
        return opt
