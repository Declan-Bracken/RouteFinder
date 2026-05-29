"""
Launched by torchrun from the Kaggle notebook for multi-GPU training.

Usage (from notebook cell):
    !torchrun --nproc_per_node=2 /kaggle/working/RouteFinder/train/scripts/ddp_launch.py

torchrun spawns one process per GPU and sets LOCAL_RANK/RANK/WORLD_SIZE before
any Python code runs. Each process uses devices=1 so PL manages exactly one GPU
and joins torchrun's process group instead of spawning its own.
"""
import pickle
import random
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    random.seed(local_rank)

    with open("/kaggle/working/cfg.pkl", "rb") as f:
        cfg = pickle.load(f)

    if world_size > 1:
        # torchrun handles process spawning — tell PL to manage exactly 1 GPU
        # per process and join the existing process group rather than spawning.
        cfg.devices = 1
        cfg.strategy = "ddp"

    from train.train import train
    train(cfg)


if __name__ == "__main__":
    main()
