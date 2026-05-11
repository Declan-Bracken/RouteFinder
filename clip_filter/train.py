"""
Fine-tune CLIP ViT-B/32 for binary route photo classification (keep / reject).

Training strategy:
  Phase 1 — head only (10 epochs):  fast convergence, encoder frozen
  Phase 2 — partial unfreeze (30 epochs): last N transformer blocks + head,
             with differential learning rates (encoder 1e-5, head 1e-4)

Loss: Asymmetric Loss — aggressive negative focusing for high precision.

Output: models/clip_filter/best.pth
"""

import torch
import argparse
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from pathlib import Path
from clip_filter.dataset import ANNOTATED_CSV, CHECKPOINT_DIR, download_and_cache, make_loaders
from clip_filter.model import build_model, evaluate
from clip_filter.loss import AsymmetricLoss
from clip_filter.eval import summary_metrics, pareto_score, print_threshold_table


def set_encoder_frozen(model, frozen: bool):
    for p in model.visual.parameters():
        p.requires_grad = not frozen


def unfreeze_last_blocks(model, n: int):
    """Unfreeze only the last N transformer blocks + post-norm + projection."""
    set_encoder_frozen(model, frozen=True)
    blocks = list(model.visual.transformer.resblocks)
    for block in blocks[-n:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.visual.ln_post.parameters():
        p.requires_grad = True
    if model.visual.proj is not None:
        model.visual.proj.requires_grad = True


def forward_pass(model, head, imgs, encoder_frozen):
    if encoder_frozen:
        with torch.no_grad():
            feats = model.encode_image(imgs).float()
    else:
        feats = model.encode_image(imgs).float()
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return head(feats).squeeze(1)


def train_epoch(model, head, loader, optimizer, criterion, device, encoder_frozen, desc="train"):
    model.train() if not encoder_frozen else model.eval()
    head.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = forward_pass(model, head, imgs, encoder_frozen)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


def save_checkpoint(model, head, val_prec, keep_perc, epoch, phase, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "head_state_dict":  head.state_dict(),
        "val_precision":    val_prec,
        "keep_percent":     keep_perc,
        "epoch":            epoch,
        "phase":            phase,
    }, path)


def run_phase(
    model, head, criterion, train_loader, val_loader,
    optimizer, n_epochs, phase, device,
    encoder_frozen, min_keep_pct, ckpt_path, best_score,
    patience=None,
):
    """Run one training phase. Returns (scores, labels, best_score)."""
    scores, labels = None, None
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(model, head, train_loader, optimizer, criterion, device,
                           encoder_frozen=encoder_frozen, desc=f"P{phase} e{epoch}")
        scores, labels = evaluate(model, head, val_loader, device)
        prec, _, _, keep_pct = summary_metrics(labels, scores, monitor_threshold=0.70)
        score, _, pareto_keep_pct = pareto_score(labels, scores, min_keep_pct=min_keep_pct)
        ckpt_marker = ""
        if score > best_score:
            best_score = score
            no_improve = 0
            save_checkpoint(model, head, score, pareto_keep_pct, epoch, phase=phase, path=ckpt_path)
            ckpt_marker = "  ✓"
        elif patience is not None:
            no_improve += 1
            ckpt_marker = f"  ({no_improve}/{patience})"
        print(f"[P{phase} {epoch:>2}/{n_epochs}]  loss={loss:.4f}  "
              f"prec@0.70={prec:.3f}  keep={keep_pct:.1f}%@0.70  "
              f"pareto={score:.3f}@keep={pareto_keep_pct:.2f}{ckpt_marker}")
        if patience is not None and no_improve >= patience:
            print(f"\nEarly stopping — no pareto improvement for {patience} epochs.")
            break

    return scores, labels, best_score


def training_loop(
    model, head, criterion, train_loader, val_loader,
    phase1_epochs=10, phase2_epochs=30,
    resume=None, device="cpu", min_keep_pct=5.0, ckpt_path=None,
    unfreeze_blocks=4, encoder_lr=1e-5, head_lr=1e-3, patience=5,
):
    best_score = 0.0
    start_phase = 1

    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        head.load_state_dict(ckpt["head_state_dict"])
        best_score = ckpt.get("val_precision", 0.0)
        start_phase = 2
        print(f"Resumed from {resume}  (epoch={ckpt['epoch']}, phase={ckpt['phase']}, score={best_score:.3f})")
        print("Skipping phase 1 — jumping straight to phase 2.")

    # ── Phase 1: head only ────────────────────────────────────────────────────
    if start_phase == 1:
        print(f"\n{'─'*50}\nPhase 1: head only ({phase1_epochs} epochs)\n{'─'*50}")
        set_encoder_frozen(model, frozen=True)
        optimizer = AdamW(head.parameters(), lr=head_lr, weight_decay=0.0)
        scores, labels, best_score = run_phase(
            model, head, criterion, train_loader, val_loader,
            optimizer, phase1_epochs, phase=1, device=device,
            encoder_frozen=True, min_keep_pct=min_keep_pct,
            ckpt_path=ckpt_path, best_score=best_score,
            patience=None,
        )
        print("\nPhase 1 final val metrics:")
        print_threshold_table(labels, scores)

    # ── Phase 2: partial unfreeze ─────────────────────────────────────────────
    print(f"\n{'─'*50}\nPhase 2: unfreeze last {unfreeze_blocks} blocks ({phase2_epochs} epochs)\n{'─'*50}")
    unfreeze_last_blocks(model, unfreeze_blocks)
    encoder_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW([
        {"params": encoder_params,    "lr": encoder_lr, "weight_decay": 1e-2},
        {"params": head.parameters(), "lr": head_lr / 10, "weight_decay": 0.0},
    ])
    scores, labels, best_score = run_phase(
        model, head, criterion, train_loader, val_loader,
        optimizer, phase2_epochs, phase=2, device=device,
        encoder_frozen=False, min_keep_pct=min_keep_pct,
        ckpt_path=ckpt_path, best_score=best_score,
        patience=patience,
    )

    return scores, labels, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated_csv",   default=str(ANNOTATED_CSV))
    parser.add_argument("--checkpoint_path", default=str(CHECKPOINT_DIR / "best.pth"))
    parser.add_argument("--train_frac",      type=float, default=0.8)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--phase1_epochs",   type=int,   default=10)
    parser.add_argument("--phase2_epochs",   type=int,   default=30)
    parser.add_argument("--unfreeze_blocks", type=int,   default=4,
                        help="Number of ViT-B/32 transformer blocks to unfreeze in phase 2 (max 12)")
    parser.add_argument("--encoder_lr",      type=float, default=1e-5)
    parser.add_argument("--head_lr",         type=float, default=1e-3)
    parser.add_argument("--gamma_neg",       type=float, default=4.0)
    parser.add_argument("--gamma_pos",       type=float, default=0.0)
    parser.add_argument("--margin",          type=float, default=0.05)
    parser.add_argument("--concurrency",     type=int,   default=30)
    parser.add_argument("--min_keep_pct",    type=float, default=5.0,
                        help="Min keep%% required when computing Pareto checkpoint score")
    parser.add_argument("--resume",          type=str,   default=None,
                        help="Path to checkpoint to resume from. Skips phase 1.")
    parser.add_argument("--patience",        type=int,   default=2,
                        help="Early stopping patience for phase 2 (epochs without pareto improvement)")
    args = parser.parse_args()

    df = pd.read_csv(args.annotated_csv)
    df["keep"] = df["keep"].astype(bool)
    print(f"Loaded {len(df)} annotations  (pos: {df['keep'].sum()}, neg: {(~df['keep']).sum()})")

    df = download_and_cache(df, args.concurrency)
    train_loader, val_loader = make_loaders(df, args.train_frac, args.batch_size)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    model, head = build_model(device)
    criterion = AsymmetricLoss(args.gamma_neg, args.gamma_pos, args.margin)

    scores, labels, best_score = training_loop(
        model, head, criterion, train_loader, val_loader,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        resume=args.resume,
        device=device,
        min_keep_pct=args.min_keep_pct,
        ckpt_path=Path(args.checkpoint_path),
        unfreeze_blocks=args.unfreeze_blocks,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        patience=args.patience,
    )

    print("\nFinal val metrics:")
    print_threshold_table(labels, scores)
    print(f"\nDone. Best pareto score: {best_score:.3f}")
    print(f"Checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
