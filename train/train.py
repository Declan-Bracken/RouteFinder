"""
Training entry point — Kaggle-portable.
To use on Kaggle: upload the routefinder/ package as a dataset input,
then run: !pip install -e /kaggle/input/routefinder-package/
"""
import argparse
from dataclasses import dataclass


@dataclass
class Config:
    model: str = "supcon"          # "simclr" | "supcon"
    backbone: str = "resnet50"
    lr: float = 1e-5
    temperature: float = 0.07
    weight_decay: float = 1e-4
    num_unfrozen_blocks: int = 1
    batch_size: int = 64
    max_epochs: int = 50
    hf_dataset: str = "DeclanBracken/RouteFinderDatasetV2"
    checkpoint_dir: str = "models/checkpoints"
    device: str = "auto"


def train(config: Config):
    raise NotImplementedError("Training script coming soon — see train/notebooks/ for current Kaggle notebooks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="supcon")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    cfg = Config(model=args.model, lr=args.lr, max_epochs=args.epochs)
    train(cfg)
