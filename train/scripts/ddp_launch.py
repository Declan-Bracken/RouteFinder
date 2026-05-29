"""
Launched by torchrun from the Kaggle notebook for multi-GPU training.

Usage (from notebook cell):
    !torchrun --nproc_per_node=2 /kaggle/working/RouteFinder/train/scripts/ddp_launch.py

torchrun spawns one process per GPU and sets RANK/LOCAL_RANK/WORLD_SIZE.
Each process runs this script with devices=1 — PL manages one GPU per
process and joins the distributed group torchrun already set up.
"""
import pickle
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    with open("/kaggle/working/cfg.pkl", "rb") as f:
        cfg = pickle.load(f)

    # torchrun already created one process per GPU — tell PL to manage 1 GPU
    # per process rather than spawning its own workers.
    cfg.devices = 1

    from train.train import train
    train(cfg)


if __name__ == "__main__":
    main()
