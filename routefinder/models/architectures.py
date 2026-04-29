import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from pytorch_metric_learning.losses import SupConLoss
from lightly.loss import NTXentLoss


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class SimCLREncoder(pl.LightningModule):
    """ResNet-50 backbone with a 2-layer projection head trained with NTXentLoss."""

    def __init__(self, lr=5e-4, temperature=0.1):
        super().__init__()
        self.encoder = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.proj = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128),
        )
        self.lr = lr
        self.temp = temperature
        self.criterion = NTXentLoss(temperature=temperature)

    def forward(self, x):
        return nn.functional.normalize(self.proj(self.encoder(x)), dim=-1)

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        loss = self.criterion(self(x1), self(x2))
        self.log("simclr_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch
        val_loss = self.criterion(self(x1), self(x2))
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)


class SupConModel(pl.LightningModule):
    """Supervised contrastive model with a partially-frozen ResNet-50 backbone."""

    def __init__(self, base=None, lr=1e-5, temperature=0.07, num_unfrozen_blocks=1, weight_decay=1e-4):
        super().__init__()
        self.encoder = base if base is not None else timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.loss_fn = SupConLoss(temperature=temperature)
        self.lr = lr
        self.weight_decay = weight_decay

        for param in self.encoder.parameters():
            param.requires_grad = False

        layers = [self.encoder.layer1, self.encoder.layer2, self.encoder.layer3, self.encoder.layer4]
        for layer in layers[-num_unfrozen_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        return nn.functional.normalize(self.encoder(x), dim=-1)

    def training_step(self, batch, idx):
        x, labels = batch
        if isinstance(labels, (tuple, list)):
            labels = torch.stack(labels)
        labels = labels.to(self.device)
        loss = self.loss_fn(self(x), labels)
        self.log("supcon_train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        x, labels = batch
        if isinstance(labels, (tuple, list)):
            labels = torch.stack(labels)
        labels = labels.to(self.device)
        loss = self.loss_fn(self(x), labels)
        self.log("supcon_val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def load_model(ckpt_path, model_class=None):
    """
    Load a Lightning checkpoint. model_class defaults to SimCLREncoder.
    Example: model = load_model("models/pretrained/simclr.ckpt")
             model = load_model("models/pretrained/supcon.ckpt", SupConModel)
    """
    if model_class is None:
        model_class = SimCLREncoder
    model = model_class.load_from_checkpoint(ckpt_path)
    model.eval()
    return model


@torch.no_grad()
def encode_batch(images, model, device="cpu"):
    """
    Encode a list of PIL Images. Returns a (N, D) tensor of L2-normalised embeddings.
    """
    batch = torch.stack([INFERENCE_TRANSFORM(img) for img in images]).to(device)
    return model(batch)
