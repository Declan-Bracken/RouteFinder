"""
Evaluate a trained model using Recall@K on a held-out set.

Metrics reported:
  - Recall@1, @5, @10  (primary signal)
  - Mean Reciprocal Rank
"""
import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from routefinder.models.architectures import load_model, encode_batch, SupConModel
from routefinder.data.download import load_image


def recall_at_k(embeddings: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    Fraction of queries where at least one of the top-k nearest neighbours
    shares the query's label (excluding the query itself).
    """
    norm = torch.nn.functional.normalize(embeddings, dim=1)
    sim = norm @ norm.T
    sim.fill_diagonal_(-1)  # exclude self

    n = len(labels)
    hits = 0
    for i in range(n):
        top_k = sim[i].topk(k).indices
        if any(labels[j] == labels[i] for j in top_k):
            hits += 1
    return hits / n


def evaluate(ckpt_path: str, eval_csv: str, image_dir: str, device: str = "cpu"):
    df = pd.read_csv(eval_csv)
    df = df[df["keep"] == True].reset_index(drop=True)

    model = load_model(ckpt_path, SupConModel)
    model = model.to(device)

    images, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        img = load_image(row["url"], image_dir)
        if img is not None:
            images.append(img.convert("RGB"))
            labels.append(row["label"])

    embeddings = encode_batch(images, model, device)
    label_tensor = torch.tensor(labels)

    for k in (1, 5, 10):
        r = recall_at_k(embeddings, label_tensor, k)
        print(f"Recall@{k}: {r:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--csv", default="data/tagged_trees/processed_mountain_project_tree.csv")
    parser.add_argument("--image_dir", default="data/cache")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate(args.ckpt, args.csv, args.image_dir, args.device)
