"""
Launched by torchrun from the Kaggle notebook for multi-GPU training.

Usage (from notebook cell):
    !torchrun --nproc_per_node=2 /kaggle/working/RouteFinder/train/scripts/ddp_launch.py

torchrun spawns one process per GPU and sets RANK/LOCAL_RANK/WORLD_SIZE.
Each process runs this script with devices=1 — PL manages one GPU per
process and joins the distributed group torchrun already set up.
"""
import pickle
import random
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    # Known benign DDP+AMP stream mismatch — no correctness impact
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    # Seed per rank so each GPU produces different batch orderings
    rank = int(os.environ.get("LOCAL_RANK", 0))
    random.seed(rank)

    with open("/kaggle/working/cfg.pkl", "rb") as f:
        cfg = pickle.load(f)

    from train.train import train
    train(cfg)


if __name__ == "__main__":
    main()
