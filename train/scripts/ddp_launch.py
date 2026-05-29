"""
Launched by `!python` from the Kaggle notebook cell.
PL's DDPStrategy(start_method="spawn") handles spawning one process per GPU.

Each spawned process inherits this script's data loaders, but the samplers
split routes by rank at __iter__ time using torch.distributed.get_rank(),
which IS available after PL initialises the process group inside each worker.
"""
import pickle
import random
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    with open("/kaggle/working/cfg.pkl", "rb") as f:
        cfg = pickle.load(f)

    from train.train import train
    train(cfg)


if __name__ == "__main__":
    main()
