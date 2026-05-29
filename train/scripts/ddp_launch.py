"""
Launched by torchrun from the Kaggle notebook for multi-GPU training.

Usage (from notebook cell):
    !torchrun --nproc_per_node=2 /kaggle/working/RouteFinder/train/scripts/ddp_launch.py

The notebook saves the Config to /kaggle/working/cfg.pkl before calling this.
torchrun spawns one process per GPU, each picks up the same config, and PL
auto-detects the distributed env vars (RANK, LOCAL_RANK, WORLD_SIZE) set by
torchrun and uses DDP without any Jupyter compatibility issues.
"""
import pickle
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

with open("/kaggle/working/cfg.pkl", "rb") as f:
    cfg = pickle.load(f)

from train.train import train
train(cfg)
