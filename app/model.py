import os
import sys
from pathlib import Path
from PIL import Image

_model = None
_transform = None
_device = None


def _ensure_checkpoint(settings) -> str:
    """Return local path to checkpoint, downloading from B2 if needed."""
    b2_key = settings.model_checkpoint_b2_key
    local_path = settings.model_checkpoint

    if local_path and os.path.exists(local_path):
        return local_path

    if not b2_key:
        raise RuntimeError(
            "No checkpoint available. Set MODEL_CHECKPOINT_B2_KEY to a B2 object key."
        )

    import boto3
    from botocore.config import Config as BotoConfig

    dest = Path("/tmp") / Path(b2_key).name
    if not dest.exists():
        print(f"Downloading checkpoint from B2: {b2_key} → {dest}")
        b2 = boto3.client(
            "s3",
            endpoint_url=settings.b2_endpoint_url,
            aws_access_key_id=settings.b2_key_id,
            aws_secret_access_key=settings.b2_application_key,
            config=BotoConfig(signature_version="s3v4"),
        )
        b2.download_file(settings.b2_bucket_name, b2_key, str(dest))
        print("Checkpoint downloaded.")

    return str(dest)


def _load():
    global _model, _transform, _device
    from .config import get_settings
    settings = get_settings()

    ckpt = _ensure_checkpoint(settings)

    train_dir = os.path.join(os.path.dirname(__file__), "..", "train")
    if train_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(train_dir))

    from train import RouteFinderModel, EVAL_TRANSFORM
    import torch

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = RouteFinderModel.load_from_checkpoint(ckpt, map_location=_device).eval()
    _transform = EVAL_TRANSFORM


def embed_image(image: Image.Image) -> list[float]:
    if _model is None:
        _load()
    import torch
    tensor = _transform(image.convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        return _model(tensor).squeeze(0).cpu().tolist()
