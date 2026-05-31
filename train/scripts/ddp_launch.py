"""
Launch multi-GPU training from the Kaggle notebook cell:

    import pickle
    with open("/kaggle/working/cfg.pkl", "wb") as f:
        pickle.dump(cfg, f)
    !python /kaggle/working/RouteFinder/train/scripts/ddp_launch.py

PL's DDPStrategy(start_method="spawn") handles all process spawning.
Do NOT use torchrun — it conflicts with PL's own spawning.
"""
import pickle
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
