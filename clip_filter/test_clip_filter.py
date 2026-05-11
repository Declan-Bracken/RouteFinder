"""
Validate prototype-based CLIP filtering against manually-annotated ground truth.

Uses data/tagged_trees/processed_mountain_project_tree.csv (keep=True/False labels
from the Streamlit annotator) to measure how well CLIP separates good route photos
from noise before committing to a full filtering run.

Approach:
  - 70% of labeled images build positive/negative prototypes (class mean embeddings)
  - Remaining 30% are scored by cosine similarity to each prototype → softmax → prob
  - Metrics reported across multiple thresholds

Outputs:
  - Precision / recall / F1 at multiple thresholds
  - Confusion matrix at the chosen threshold
  - data/clip_validation_results.csv  (test-set images with scores)

Usage:
    pip install open-clip-torch aiohttp scikit-learn
    python scripts/test_clip_filter.py            # balanced 200-image sample
    python scripts/test_clip_filter.py --sample 0 # all 2851 annotated images
"""

import argparse
import asyncio
import io
from pathlib import Path

import aiohttp
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ANNOTATED_CSV = Path("data/tagged_trees/processed_mountain_project_tree.csv")
OUT_CSV       = Path("data/clip_validation_results.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RouteFinder-research/1.0)"}


# ── Download ──────────────────────────────────────────────────────────────────

async def _fetch(session, url, sem):
    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as r:
                    if r.status == 200:
                        return await r.read()
                    if r.status in (403, 404, 410):
                        return None
            except Exception:
                pass
            await asyncio.sleep(1.5 ** attempt)
    return None


async def download_all(urls, concurrency=30):
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrency)) as session:
        return await asyncio.gather(*[_fetch(session, u, sem) for u in urls])


# ── CLIP ──────────────────────────────────────────────────────────────────────

def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def encode_images(images, model, preprocess, device, batch_size=32):
    """Returns (N, D) L2-normalized embedding tensor."""
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(img) for img in images[i:i+batch_size]]).to(device)
        embs = model.encode_image(batch)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)  # (N, D)


def compute_prototypes(embeddings, labels):
    """
    embeddings: (N, D) float tensor
    labels:     list/array of bool  (True = keep)
    Returns (pos_proto, neg_proto), each L2-normalized (D,) tensor.
    """
    labels = torch.tensor(labels, dtype=torch.bool)
    pos_proto = embeddings[labels].mean(dim=0)
    neg_proto = embeddings[~labels].mean(dim=0)
    return pos_proto / pos_proto.norm(), neg_proto / neg_proto.norm()


def score_with_prototypes(embeddings, pos_proto, neg_proto):
    """
    Returns list of floats: P(keep) for each embedding,
    computed as softmax([cos_sim(e, pos), cos_sim(e, neg)])[0].
    """
    protos = torch.stack([pos_proto, neg_proto])  # (2, D)
    sims = embeddings @ protos.T                  # (N, 2) — dot product of unit vecs = cosine sim
    probs = sims.softmax(dim=-1)
    return probs[:, 0].tolist()                   # positive class probability


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_threshold_table(y_true, scores):
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    print(f"\n{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Keep%':>7}")
    print("-" * 52)
    for t in thresholds:
        y_pred = [s >= t for s in scores]
        tp = sum(a and b for a, b in zip(y_true, y_pred))
        fp = sum(not a and b for a, b in zip(y_true, y_pred))
        fn = sum(a and not b for a, b in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        keep_pct = 100 * sum(y_pred) / len(y_pred)
        print(f"{t:>10.2f}  {prec:>10.3f}  {rec:>8.3f}  {f1:>8.3f}  {keep_pct:>6.1f}%")


def print_confusion(y_true, y_pred, threshold):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix at threshold={threshold:.2f}:")
    print(f"  True positives  (correctly kept):     {tp}")
    print(f"  True negatives  (correctly rejected):  {tn}")
    print(f"  False positives (noise let through):   {fp}")
    print(f"  False negatives (good photos dropped): {fn}")


def print_worst_errors(test_df, threshold, n=10):
    df = test_df[test_df["clip_score"].notna()].copy()
    df["pred"] = df["clip_score"] >= threshold
    fp = df[df["pred"] & ~df["keep"]].nlargest(n // 2, "clip_score")
    fn = df[~df["pred"] & df["keep"]].nsmallest(n // 2, "clip_score")
    if not fp.empty:
        print(f"\nTop false positives (CLIP kept, human rejected):")
        for _, row in fp.iterrows():
            print(f"  score={row['clip_score']:.3f}  {row['url']}")
    if not fn.empty:
        print(f"\nTop false negatives (CLIP rejected, human kept):")
        for _, row in fn.iterrows():
            print(f"  score={row['clip_score']:.3f}  {row['url']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated_csv", default=str(ANNOTATED_CSV))
    parser.add_argument("--out",           default=str(OUT_CSV))
    parser.add_argument("--sample", type=int, default=200,
                        help="Images to evaluate (0 = all). Balanced by class when sampling.")
    parser.add_argument("--train_frac", type=float, default=0.7,
                        help="Fraction used to build prototypes (rest is test set).")
    parser.add_argument("--threshold",   type=float, default=0.55)
    parser.add_argument("--concurrency", type=int, default=30)
    args = parser.parse_args()

    # ── Load & optionally sample ──────────────────────────────────────────────
    df = pd.read_csv(args.annotated_csv)
    df["keep"] = df["keep"].astype(bool)
    print(f"Loaded {len(df)} annotated images  "
          f"(keep=True: {df['keep'].sum()}, False: {(~df['keep']).sum()})")

    if args.sample > 0:
        n_each = args.sample // 2
        pos = df[df["keep"]].sample(min(n_each, df["keep"].sum()), random_state=42)
        neg = df[~df["keep"]].sample(min(n_each, (~df["keep"]).sum()), random_state=42)
        df = pd.concat([pos, neg]).reset_index(drop=True)
        print(f"Sampled {len(df)} images ({len(pos)} pos, {len(neg)} neg)")

    # ── Stratified train/test split ───────────────────────────────────────────
    # Do this before downloading so the split is known regardless of download failures.
    rng = np.random.default_rng(42)
    df["split"] = "test"
    for keep_val in [True, False]:
        idx = df[df["keep"] == keep_val].index
        n_train = int(len(idx) * args.train_frac)
        train_idx = rng.choice(idx, size=n_train, replace=False)
        df.loc[train_idx, "split"] = "train"

    n_train = (df["split"] == "train").sum()
    n_test  = (df["split"] == "test").sum()
    print(f"Split: {n_train} train (prototype), {n_test} test (evaluation)")

    # ── Download ──────────────────────────────────────────────────────────────
    print(f"\nDownloading {len(df)} images ({args.concurrency} concurrent)…")
    raw = asyncio.run(download_all(df["url"].tolist(), args.concurrency))

    # Build parallel lists: images[k] ↔ df.iloc[valid_idx[k]]
    images, valid_idx = [], []
    for i, data in enumerate(raw):
        if data is None:
            continue
        try:
            images.append(Image.open(io.BytesIO(data)).convert("RGB"))
            valid_idx.append(i)
        except Exception:
            pass

    print(f"Downloaded {len(images)} OK, {len(df) - len(images)} failed/404")

    # ── Encode all downloaded images ──────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\nLoading CLIP ViT-B/32 on {device}…")
    model, preprocess = load_clip(device)

    print(f"Encoding {len(images)} images…")
    all_embs = encode_images(images, model, preprocess, device)  # (N_downloaded, D)

    # Attach embeddings back to df rows using valid_idx as the bridge.
    # valid_idx[k] is the df row that images[k] came from.
    # We use a sub-dataframe (downloaded_df) that carries an _emb_idx column
    # pointing into all_embs, so the label/split/embedding mapping stays tight.
    downloaded_df = df.iloc[valid_idx].copy()
    downloaded_df["_emb_idx"] = range(len(downloaded_df))

    train_rows = downloaded_df[downloaded_df["split"] == "train"]
    test_rows  = downloaded_df[downloaded_df["split"] == "test"]

    train_embs   = all_embs[train_rows["_emb_idx"].tolist()]
    train_labels = train_rows["keep"].tolist()
    test_embs    = all_embs[test_rows["_emb_idx"].tolist()]
    test_labels  = test_rows["keep"].tolist()

    print(f"  Train: {len(train_rows)} ({sum(train_labels)} pos, {sum(not l for l in train_labels)} neg)")
    print(f"  Test:  {len(test_rows)}  ({sum(test_labels)} pos, {sum(not l for l in test_labels)} neg)")

    # ── Prototype classifier ──────────────────────────────────────────────────
    print("\nComputing prototypes…")
    pos_proto, neg_proto = compute_prototypes(train_embs, train_labels)
    proto_scores = score_with_prototypes(test_embs, pos_proto, neg_proto)

    # ── Logistic regression on frozen CLIP features ───────────────────────────
    print("Fitting logistic regression on CLIP features…")
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    clf.fit(train_embs.numpy(), train_labels)
    lr_scores = clf.predict_proba(test_embs.numpy())[:, 1].tolist()

    # ── Save & report both ────────────────────────────────────────────────────
    test_df = test_rows.drop(columns=["_emb_idx"]).copy()
    test_df["proto_score"] = [round(s, 4) for s in proto_scores]
    test_df["lr_score"]    = [round(s, 4) for s in lr_scores]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(args.out, index=False)
    print(f"Results saved to {args.out}")

    base_rate = 100 * sum(test_labels) / len(test_labels)
    print(f"\n=== Prototype (test n={len(test_df)}, base rate={base_rate:.1f}%) ===")
    print_threshold_table(test_labels, proto_scores)

    print(f"\n=== Logistic Regression on CLIP features (test n={len(test_df)}, base rate={base_rate:.1f}%) ===")
    print_threshold_table(test_labels, lr_scores)

    print(f"\n--- Confusion matrix at threshold={args.threshold:.2f} ---")
    print("Prototype:")
    y_pred_proto = [s >= args.threshold for s in proto_scores]
    print_confusion(test_labels, y_pred_proto, args.threshold)
    print("Logistic Regression:")
    y_pred_lr = [s >= args.threshold for s in lr_scores]
    print_confusion(test_labels, y_pred_lr, args.threshold)
    print(classification_report(test_labels, y_pred_lr, target_names=["reject", "keep"]))

    if "label" in test_df.columns:
        kept = test_df[test_df["lr_score"] >= args.threshold]
        per_route = kept.groupby("label").size()
        viable = (per_route >= 2).sum()
        total_routes = test_df["label"].nunique()
        print(f"\nRoute-level estimate at threshold={args.threshold:.2f}:")
        print(f"  Routes with ≥2 images after filter: {viable} / {total_routes} "
              f"({100*viable/total_routes:.1f}%)")


if __name__ == "__main__":
    main()
