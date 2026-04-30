"""
Lazy singleton model loader for inference.

The model is loaded on the first call to embed_image() and reused for
all subsequent requests — loading DINOv2 on every request would be too slow.
"""
import os
import sys
import torch
from PIL import Image

_model = None
_transform = None
_device = None


def _load():
    global _model, _transform, _device
    from .config import get_settings
    ckpt = get_settings().model_checkpoint
    if not ckpt:
        raise RuntimeError(
            "MODEL_CHECKPOINT is not set. "
            "Train a model and set the env var to its path or HF model id."
        )

    train_dir = os.path.join(os.path.dirname(__file__), "..", "train")
    if train_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(train_dir))

    from train import RouteFinderModel, EVAL_TRANSFORM

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = RouteFinderModel.load_from_checkpoint(ckpt, map_location=_device).eval()
    _transform = EVAL_TRANSFORM


def embed_image(image: Image.Image) -> list[float]:
    if _model is None:
        _load()
    tensor = _transform(image.convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        return _model(tensor).squeeze(0).cpu().tolist()
