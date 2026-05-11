"""
Evaluate a saved CLIP filter checkpoint on the annotated val set.

Usage:
    python -m clip_filter.eval
    python -m clip_filter.eval --checkpoint models/clip_filter/best.pth
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from clip_filter.dataset import ANNOTATED_CSV, RoutePhotoDataset, VAL_TRANSFORM, _cache_path, make_val_split
from clip_filter.model import load_model, evaluate as run_evaluate


def _threshold_stats(y_true, scores, t):
    y_pred = [s >= t for s in scores]
    tp = sum(a and b for a, b in zip(y_true, y_pred))
    fp = sum(not a and b for a, b in zip(y_true, y_pred))
    fn = sum(a and not b for a, b in zip(y_true, y_pred))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    keep_pct = 100 * sum(y_pred) / len(y_pred)
    return prec, rec, f1, keep_pct


def summary_metrics(y_true, scores, monitor_threshold=0.70):
    """Single-line summary for per-epoch printing."""
    prec, rec, f1, keep_pct = _threshold_stats(y_true, scores, monitor_threshold)
    return prec, rec, f1, keep_pct


def pareto_score(y_true, scores, min_keep_pct=5.0, thresholds=None):
    """
    Max precision over all thresholds where keep% >= min_keep_pct.
    Saves the model with the best precision that still retains enough data.
    """
    if thresholds is None:
        thresholds = [t / 100 for t in range(40, 99)]
    best = 0.0
    best_t = 0.0
    best_keep_pct = 0.0
    for t in thresholds:
        prec, _, _, keep_pct = _threshold_stats(y_true, scores, t)
        if keep_pct >= min_keep_pct and prec > best:
            best, best_t, best_keep_pct = prec, t, keep_pct
    if best == 0.0:
        print(f"Warning: no threshold achieved keep% >= {min_keep_pct:.1f}% — pareto score is 0. "
              f"Try lowering --min_keep_pct.")
    return best, best_t, best_keep_pct


def print_threshold_table(y_true, scores, thresholds=None):
    if thresholds is None:
        thresholds = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90,
                      0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    print(f"\n{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Keep%':>7}")
    print("-" * 52)
    best_prec, best_t = 0.0, 0.0
    for t in thresholds:
        prec, rec, f1, keep_pct = _threshold_stats(y_true, scores, t)
        print(f"{t:>10.2f}  {prec:>10.3f}  {rec:>8.3f}  {f1:>8.3f}  {keep_pct:>6.1f}%")
        if prec > best_prec:
            best_prec, best_t = prec, t
    return best_prec, best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    default="models/clip_filter/best.pth")
    parser.add_argument("--annotated_csv", default=str(ANNOTATED_CSV))
    parser.add_argument("--train_frac",    type=float, default=0.8)
    parser.add_argument("--batch_size",    type=int,   default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.annotated_csv)
    df["keep"] = df["keep"].astype(bool)
    df["cache_path"] = df["url"].apply(lambda u: str(_cache_path(u)))
    df = df[df["cache_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    df = make_val_split(df, args.train_frac)
    val_df = df[df["split"] == "val"]
    val_records = [(Path(r.cache_path), int(r.keep)) for r in val_df.itertuples()]
    val_loader = DataLoader(
        RoutePhotoDataset(val_records, VAL_TRANSFORM),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
    )
    print(f"Val set: {len(val_df)} images ({val_df['keep'].sum()} pos)")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model, head, ckpt = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (phase {ckpt['phase']}, "
          f"val_precision={ckpt['val_precision']:.3f})")

    scores, labels = run_evaluate(model, head, val_loader, device)

    print("\n=== Best checkpoint — full threshold table ===")
    print_threshold_table(labels, scores, thresholds=[
        0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
        0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
    ])


if __name__ == "__main__":
    main()
