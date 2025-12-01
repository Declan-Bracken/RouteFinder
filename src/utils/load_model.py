import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
from lightly.loss import NTXentLoss 
import torchvision.transforms as T
from PIL import Image
import os

class encoder(pl.LightningModule):
    def __init__(self, lr=5e-4, temperature=0.1):
        super().__init__()
        self.encoder = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.proj = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128)
        )
        self.lr = lr
        self.temp = temperature
        self.criterion = NTXentLoss(temperature=temperature)

    def forward(self, x):
        h = self.encoder(x)
        z = self.proj(h)
        return nn.functional.normalize(z, dim=-1)

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        z1 = self(x1)
        z2 = self(x2)
        loss = self.criterion(z1, z2)
        self.log("simclr_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch
        z1, z2 = self(x1), self(x2)
        val_loss = self.criterion(z1, z2)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

def load_model(ckpt_path):
    # model = encoder()
    model = encoder.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

@torch.no_grad()
def encode_batch(batch, model, device):
    """
    Batch is a list of PIL Image objects
    """
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),   # keep aspect ratio
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    batch = torch.stack([transform(image) for image in batch], dim = 0)
    batch = batch.to(device)
    outputs = model(batch)
    return outputs

def load_images_from_dir(directory):
    imgs = []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for fname in os.listdir(directory):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_ext:
            path = os.path.join(directory, fname)
            try:
                imgs.append(Image.open(path).convert("RGB"))
            except Exception as e:
                print(f"Warning: skipping corrupted file {path} ({e})")
    return imgs

if __name__ == "__main__":
    path = "models/pretrained/simclr.ckpt"
    image_dir = "src/data/cache"
    device = "cpu"
    model = load_model(path)
    print("Model succesfully loaded")
    
    batch = load_images_from_dir(image_dir)
    encoded_batch = encode_batch(batch, model, device)
