"""
Visualize images kept/rejected by the CLIP filter at a given threshold.

Generates an HTML file showing scored images so you can qualitatively assess
whether the model is making sensible decisions, independent of the labels.

Border colors:
  green  — model keeps, human labeled keep    (true positive)
  orange — model keeps, human labeled reject  (false positive — worth examining)
  red    — model rejects, human labeled keep  (false negative — spot check)

Usage:
    python -m clip_filter.visualize
    python -m clip_filter.visualize --threshold 0.97 --show_fn 20
    open data/clip_filter_viz.html
"""

import argparse
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from clip_filter.dataset import (
    ANNOTATED_CSV, RoutePhotoDataset, VAL_TRANSFORM, _cache_path, make_val_split,
)
from clip_filter.model import load_model, evaluate

OUT_HTML = Path("data/clip_filter_viz.html")
THUMB_SIZE = 224


def img_to_b64(path: Path, size=THUMB_SIZE) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def build_html(sections: list[dict], threshold: float, stats: dict) -> str:
    def card(rec):
        border = {
            "tp": "#16a34a",
            "fp": "#f97316",
            "fn": "#dc2626",
            "tn": "#9ca3af",
        }[rec["kind"]]
        label_text  = "✓ keep" if rec["human_keep"] else "✗ reject"
        label_color = "#16a34a" if rec["human_keep"] else "#dc2626"
        return f"""
        <div style="display:inline-block;margin:6px;vertical-align:top;
                    border:3px solid {border};border-radius:8px;overflow:hidden;
                    width:{THUMB_SIZE}px;background:#f9fafb;">
          <img src="data:image/jpeg;base64,{rec['b64']}"
               style="width:{THUMB_SIZE}px;height:{THUMB_SIZE}px;object-fit:cover;display:block;">
          <div style="padding:6px;font-family:monospace;font-size:11px;">
            <div style="font-weight:700;">score={rec['score']:.4f}</div>
            <div style="color:{label_color};">{label_text}</div>
            <div style="color:#6b7280;overflow:hidden;white-space:nowrap;
                        text-overflow:ellipsis;" title="{rec['route_name']}">
              {rec['route_name'][:28]}
            </div>
          </div>
        </div>"""

    html_sections = ""
    for sec in sections:
        cards = "".join(card(r) for r in sec["records"])
        html_sections += f"""
        <h2 style="font-family:sans-serif;margin:24px 0 8px;">
          {sec['title']} <span style="color:#6b7280;font-size:14px;">({len(sec['records'])} images)</span>
        </h2>
        <div>{cards}</div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>CLIP Filter Visualization — t={threshold}</title></head>
<body style="margin:24px;background:#fff;">
<h1 style="font-family:sans-serif;">CLIP Filter Visualization</h1>
<p style="font-family:monospace;color:#6b7280;">
  threshold={threshold} &nbsp;|&nbsp;
  val_size={stats['total']} &nbsp;|&nbsp;
  kept={stats['kept']} ({stats['kept_pct']:.1f}%) &nbsp;|&nbsp;
  tp={stats['tp']} &nbsp;fp={stats['fp']} &nbsp;fn_shown={stats['fn_shown']}
</p>
<p style="font-family:sans-serif;font-size:13px;">
  <span style="color:#16a34a;">■</span> true positive &nbsp;
  <span style="color:#f97316;">■</span> false positive (model keeps, you labeled reject) &nbsp;
  <span style="color:#dc2626;">■</span> false negative sample (model rejects, you labeled keep)
</p>
{html_sections}
</body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    default="models/clip_filter/best.pth")
    parser.add_argument("--annotated_csv", default=str(ANNOTATED_CSV))
    parser.add_argument("--train_frac",    type=float, default=0.8)
    parser.add_argument("--threshold",     type=float, default=0.97)
    parser.add_argument("--show_fn",       type=int,   default=20,
                        help="How many false negatives to show at the bottom")
    parser.add_argument("--out",           default=str(OUT_HTML))
    parser.add_argument("--batch_size",    type=int,   default=32)
    args = parser.parse_args()

    # ── Build val split (must match training split exactly) ───────────────────
    df = pd.read_csv(args.annotated_csv)
    df["keep"] = df["keep"].astype(bool)
    df["cache_path"] = df["url"].apply(lambda u: str(_cache_path(u)))
    df = df[df["cache_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    df = make_val_split(df, args.train_frac)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    print(f"Val set: {len(val_df)} images ({val_df['keep'].sum()} pos, {(~val_df['keep']).sum()} neg)")

    val_records = [(Path(r.cache_path), int(r.keep)) for r in val_df.itertuples()]
    val_loader = DataLoader(
        RoutePhotoDataset(val_records, VAL_TRANSFORM),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model, head, ckpt = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint (epoch={ckpt['epoch']}, phase={ckpt['phase']})")

    scores, labels = evaluate(model, head, val_loader, device)
    val_df["score"] = scores

    # ── Classify ──────────────────────────────────────────────────────────────
    t = args.threshold
    kept      = val_df[val_df["score"] >= t].sort_values("score", ascending=False)
    fn_sample = (val_df[(val_df["score"] < t) & val_df["keep"]]
                 .sort_values("score", ascending=False)
                 .head(args.show_fn))
    tp = kept[kept["keep"]]
    fp = kept[~kept["keep"]]

    print(f"\nAt threshold={t}:")
    print(f"  Kept:  {len(kept)} ({100*len(kept)/len(val_df):.1f}%)")
    print(f"  TP: {len(tp)}  FP: {len(fp)}  Precision: {len(tp)/max(len(kept),1):.3f}")
    print(f"  Showing {len(fn_sample)} false negatives")
    print("Encoding thumbnails…")

    def make_records(rows, kind):
        return [
            {
                "b64":        img_to_b64(Path(r.cache_path)),
                "score":      r.score,
                "human_keep": r.keep,
                "route_name": getattr(r, "route_name", str(r.Index)),
                "kind":       kind,
            }
            for r in rows.itertuples()
        ]

    sections = [
        {
            "title":   f"✓ True positives — model keeps, you kept (score ≥ {t})",
            "records": make_records(tp, "tp"),
        },
        {
            "title":   f"⚠ False positives — model keeps, you rejected (score ≥ {t})",
            "records": make_records(fp, "fp"),
        },
        {
            "title":   f"✗ False negative sample — model rejected, you kept (top {args.show_fn} by score)",
            "records": make_records(fn_sample, "fn"),
        },
    ]
    sections = [s for s in sections if s["records"]]

    stats = {
        "total": len(val_df), "kept": len(kept),
        "kept_pct": 100 * len(kept) / len(val_df),
        "tp": len(tp), "fp": len(fp), "fn_shown": len(fn_sample),
    }

    html = build_html(sections, t, stats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"\nSaved to {args.out}")
    print(f"Open with:  open {args.out}")


if __name__ == "__main__":
    main()
