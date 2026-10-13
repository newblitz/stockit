"""Improved training script for the 17-feat HierarchicalCoAttentionStockPredictor.

Key improvements over train_17feat.py (baseline):
  1. Gradient clipping (max_norm=1.0)       -- was entirely absent in baseline
  2. AdamW + weight_decay=1e-3              -- explicit L2 regularisation
  3. CosineAnnealingLR                     -- replaces near-flat StepLR (gamma=0.9999)
  4. dropout=0.3                            -- up from 0.2 to reduce overfitting
  5. Label-smoothed loss (eps=0.05)         -- softer targets for noisy stock labels
  6. epochs=40, patience=12                 -- longer budget, more patience
  7. AMP (fp16) on GPU                      -- 1.5-2x faster, same numerics
  8. Val-set threshold optimisation         -- sweep [0.30,0.70] on val, apply to test

Core architecture (model.py) is UNCHANGED.
Evaluation protocol: test split touched exactly once at the end (identical to baseline).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config_17feat import Config17Feature
from src.data_17feat import CMINWindowDataset17Feat
from src.model import HierarchicalCoAttentionStockPredictor


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _confusion(logits: Tensor, labels: Tensor, threshold: float = 0.5):
    probs = logits.sigmoid().flatten()
    truth = labels.long().flatten()
    pred  = (probs >= threshold).long()
    tp = int(((pred == 1) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    return tp, tn, fp, fn


def _mcc_acc(tp, tn, fp, fn):
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc   = 0.0 if denom == 0 else (tp * tn - fp * fn) / denom ** 0.5
    acc   = (tp + tn) / max(tp + tn + fp + fn, 1)
    return mcc, acc


def metrics_from_logits(logits: Tensor, labels: Tensor, threshold: float = 0.5) -> dict:
    tp, tn, fp, fn = _confusion(logits, labels, threshold)
    mcc, acc = _mcc_acc(tp, tn, fp, fn)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"accuracy": acc, "mcc": mcc, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Label-smoothed BCE loss
# ---------------------------------------------------------------------------

class LabelSmoothingBCELoss(nn.Module):
    """BCE-with-logits + one-sided label smoothing."""

    def __init__(self, smoothing: float = 0.05) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        smooth_targets = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, smooth_targets)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float = 0.5,
    use_amp: bool = False,
) -> tuple[dict, Tensor, Tensor]:
    model.eval()
    logit_list, label_list, loss_sum, n = [], [], 0.0, 0
    for prices, text, labels, _ in loader:
        prices, text, labels_dev = prices.to(device), text.to(device), labels.to(device)
        with autocast(enabled=use_amp):
            logits = model(prices, text)
            loss   = criterion(logits, labels_dev)
        loss_sum += loss.item() * len(labels)
        n        += len(labels)
        logit_list.append(logits.float().cpu())
        label_list.append(labels)
    all_logits = torch.cat(logit_list)
    all_labels = torch.cat(label_list)
    result = metrics_from_logits(all_logits, all_labels, threshold)
    result["loss"] = loss_sum / max(n, 1)
    return result, all_logits, all_labels


# ---------------------------------------------------------------------------
# Threshold optimisation (val set only — never touches test)
# ---------------------------------------------------------------------------

def optimise_threshold(logits: Tensor, labels: Tensor) -> tuple[float, float, float]:
    """Return (best_threshold, best_val_mcc, best_val_acc)."""
    best_mcc, best_thr, best_acc = -2.0, 0.5, 0.0
    for thr_i in range(30, 71):
        thr = thr_i / 100.0
        tp, tn, fp, fn = _confusion(logits, labels, thr)
        mcc, acc = _mcc_acc(tp, tn, fp, fn)
        if mcc > best_mcc:
            best_mcc, best_thr, best_acc = mcc, thr, acc
    return best_thr, best_mcc, best_acc


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler,
                    epoch, config, result, history, threshold=0.5):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "config": asdict(config),
            "text_embedding_dim": 1024,   # actual dim (config default is 768, a known mismatch)
            "price_dim": 17,
            "validation": result,
            "history": history,
            "threshold": threshold,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Tee (stdout → terminal + file)
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, stream, log_file: Path) -> None:
        self.stream = stream
        self.file   = log_file.open("a", encoding="utf-8", buffering=1)

    def write(self, msg: str) -> int:
        self.stream.write(msg)
        return self.file.write(msg)

    def flush(self):
        self.stream.flush()
        self.file.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Improved 17-feat co-attention trainer.")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/kaggle/working/data/CMIN-Dataset-official/CMIN-US"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("/kaggle/working/stockit/new_cache/qwen3_8b_bge_m3"))
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("checkpoints/improved_v1"))
    parser.add_argument("--epochs",          type=int,   default=40)
    parser.add_argument("--patience",        type=int,   default=12)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight-decay",    type=float, default=1e-3)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--grad-clip",       type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--eta-min",         type=float, default=1e-6)
    parser.add_argument("--no-amp",          action="store_true",
                        help="Disable automatic mixed precision (AMP).")
    parser.add_argument("--max-stocks",      type=int,
                        help="Use a small ticker subset for a smoke run.")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--log-file",        type=Path)
    parser.add_argument("--tensorboard-dir", type=Path,
                        default=Path("runs/improved_v1"))
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    args = parser.parse_args()

    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = Tee(sys.stdout, args.log_file)  # type: ignore[assignment]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # ── Config ──────────────────────────────────────────────────────────────
    config          = Config17Feature()
    config.epochs   = args.epochs
    config.patience = args.patience
    config.dropout  = args.dropout
    config.lr       = args.lr

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_set = CMINWindowDataset17Feat(args.dataset_root, args.cache_root, "train",
                                        seq_len=config.seq_len, max_stocks=args.max_stocks)
    val_set   = CMINWindowDataset17Feat(args.dataset_root, args.cache_root, "val",
                                        seq_len=config.seq_len, max_stocks=args.max_stocks)
    test_set  = CMINWindowDataset17Feat(args.dataset_root, args.cache_root, "test",
                                        seq_len=config.seq_len, max_stocks=args.max_stocks)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size*2, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size*2, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    # ── Model ────────────────────────────────────────────────────────────────
    model = HierarchicalCoAttentionStockPredictor(
        text_embedding_dim=train_set.text_embedding_dim,
        seq_len=config.seq_len, patch_len=config.patch_len, stride=config.stride,
        price_dim=train_set.price_dim,
        d_model=config.d_model, d_ff=config.d_ff,
        n_heads=config.n_heads, n_layers=config.n_layers,
        n_fusion_layers=config.n_fusion_layers,
        n_classes=config.n_classes,
        dropout=args.dropout,
    ).to(device)

    total_p    = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}  |  AMP: {use_amp}", flush=True)
    print(f"Model params: {total_p:,} total, {trainable:,} trainable", flush=True)
    print(f"Dataset: train={len(train_set):,}  val={len(val_set):,}  test={len(test_set):,}", flush=True)
    print(f"price_dim={train_set.price_dim}  text_dim={train_set.text_embedding_dim}", flush=True)
    print(
        f"Hyperparams: epochs={args.epochs}, patience={args.patience}, "
        f"lr={args.lr}, wd={args.weight_decay}, dropout={args.dropout}, "
        f"clip={args.grad_clip}, label_smooth={args.label_smoothing}, "
        f"batch={args.batch_size}",
        flush=True,
    )

    # ── Optimiser, scheduler, scaler ─────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )
    criterion = LabelSmoothingBCELoss(smoothing=args.label_smoothing)
    scaler    = GradScaler(enabled=use_amp)

    # ── Checkpoint dirs ───────────────────────────────────────────────────────
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    periodic_dir = args.checkpoint_dir / "periodic"
    if args.checkpoint_interval > 0:
        periodic_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = None
    if args.tensorboard_dir and str(args.tensorboard_dir) != "":
        args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(args.tensorboard_dir))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mcc_opt = float("-inf")
    best_val_acc     = float("-inf")
    best_threshold   = 0.5
    stale_epochs     = 0
    history: list[dict] = []
    last_ckpt = args.checkpoint_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_logits_list, train_labels_list, train_losses = [], [], []

        for prices, text, labels, _ in train_loader:
            prices, text, labels_dev = prices.to(device), text.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                logits = model(prices, text)
                loss   = criterion(logits, labels_dev)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # ── gradient clipping ──────────────────────────────────────────
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_logits_list.append(logits.float().detach().cpu())
            train_labels_list.append(labels)

        train_m = metrics_from_logits(
            torch.cat(train_logits_list), torch.cat(train_labels_list)
        )
        val_result, val_logits, val_labels = evaluate(
            model, val_loader, device, criterion, use_amp=use_amp
        )

        # ── val-only threshold search ──────────────────────────────────────
        epoch_thr, epoch_mcc_opt, epoch_acc_opt = optimise_threshold(val_logits, val_labels)

        elapsed = time.perf_counter() - t0
        result = {
            "epoch":           epoch,
            "elapsed_s":       round(elapsed, 2),
            "lr":              optimizer.param_groups[0]["lr"],
            "train_loss":      float(np.mean(train_losses)),
            "train_accuracy":  train_m["accuracy"],
            "train_mcc":       train_m["mcc"],
            "train_f1":        train_m["f1"],
            "val_loss":        val_result["loss"],
            "val_accuracy":    val_result["accuracy"],
            "val_mcc":         val_result["mcc"],       # at 0.5
            "val_f1":          val_result["f1"],
            "val_mcc_opt":     epoch_mcc_opt,           # at optimised thr
            "val_acc_opt":     epoch_acc_opt,
            "val_threshold":   epoch_thr,
        }
        history.append(result)

        if tb_writer:
            for k in ("train_loss","train_accuracy","train_mcc",
                      "val_loss","val_accuracy","val_mcc","val_mcc_opt"):
                tb_writer.add_scalar(k, result[k], epoch)
            tb_writer.add_scalar("lr", result["lr"], epoch)
            tb_writer.flush()

        # ── checkpointing ─────────────────────────────────────────────────
        ckpt_kwargs = dict(model=model, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, epoch=epoch, config=config,
                           result=result, history=history)
        save_checkpoint(last_ckpt, **ckpt_kwargs, threshold=epoch_thr)
        tags = []

        if epoch_mcc_opt > best_val_mcc_opt:
            best_val_mcc_opt = epoch_mcc_opt
            best_threshold   = epoch_thr
            stale_epochs     = 0
            save_checkpoint(args.checkpoint_dir / "best.pt",
                            **ckpt_kwargs, threshold=epoch_thr)
            tags.append(f"best.pt (val_mcc_opt={best_val_mcc_opt:.4f} thr={epoch_thr:.2f})")
        else:
            stale_epochs += 1

        if val_result["accuracy"] > best_val_acc:
            best_val_acc = val_result["accuracy"]
            save_checkpoint(args.checkpoint_dir / "best_accuracy.pt",
                            **ckpt_kwargs, threshold=epoch_thr)
            tags.append("best_accuracy.pt")

        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            save_checkpoint(periodic_dir / f"epoch_{epoch:03d}.pt",
                            **ckpt_kwargs, threshold=epoch_thr)
            tags.append(f"periodic/epoch_{epoch:03d}.pt")

        (args.checkpoint_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        with (args.checkpoint_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(result))
            w.writeheader(); w.writerows(history)

        print(
            f"Ep {epoch:03d}/{args.epochs} | {elapsed:.1f}s | "
            f"tr loss {result['train_loss']:.4f} acc {result['train_accuracy']:.4f} "
            f"mcc {result['train_mcc']:.4f} | "
            f"val loss {result['val_loss']:.4f} acc {result['val_accuracy']:.4f} "
            f"mcc@0.50 {result['val_mcc']:.4f} "
            f"mcc_opt {result['val_mcc_opt']:.4f}@{result['val_threshold']:.2f} | "
            f"lr {result['lr']:.2e}"
            + (" | " + ", ".join(tags) if tags else ""),
            flush=True,
        )

        scheduler.step()

        if config.patience > 0 and stale_epochs >= config.patience:
            print(
                f"Early stopping: {config.patience} stale epochs. "
                f"Best val_mcc_opt={best_val_mcc_opt:.4f} @ thr={best_threshold:.2f}",
                flush=True,
            )
            break

    # ── Final test evaluation (one shot, after val-driven model selection) ──
    best_path = args.checkpoint_dir / "best.pt"
    if best_path.exists():
        saved = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        best_threshold = saved.get("threshold", best_threshold)
        print(
            f"\nLoaded best checkpoint epoch={saved['epoch']} "
            f"val_mcc_opt={saved['validation']['val_mcc_opt']:.4f} "
            f"threshold={best_threshold:.2f}",
            flush=True,
        )

    test_50,  _, _               = evaluate(model, test_loader, device, criterion,
                                            threshold=0.50, use_amp=use_amp)
    test_opt, test_logits, test_labels = evaluate(model, test_loader, device, criterion,
                                                   threshold=best_threshold, use_amp=use_amp)

    print(
        f"\n{'='*70}\n"
        f"BASELINE  Test @0.50 : ACC={0.4914:.4f}  MCC={-0.0127:.4f}  "
        f"F1={0.4872:.4f}\n"
        f"IMPROVED  Test @0.50 : ACC={test_50['accuracy']:.4f}  "
        f"MCC={test_50['mcc']:.4f}  F1={test_50['f1']:.4f}\n"
        f"IMPROVED  Test @{best_threshold:.2f}: ACC={test_opt['accuracy']:.4f}  "
        f"MCC={test_opt['mcc']:.4f}  F1={test_opt['f1']:.4f}  "
        f"(val-optimised threshold)\n"
        f"Confusion @{best_threshold:.2f}: "
        f"TP={test_opt['tp']} TN={test_opt['tn']} "
        f"FP={test_opt['fp']} FN={test_opt['fn']}\n"
        f"{'='*70}",
        flush=True,
    )

    final = {
        **history[-1],
        "best_val_mcc_opt":  best_val_mcc_opt,
        "best_threshold":    best_threshold,
        "test_loss_50":      test_50["loss"],
        "test_accuracy_50":  test_50["accuracy"],
        "test_mcc_50":       test_50["mcc"],
        "test_f1_50":        test_50["f1"],
        "test_accuracy_opt": test_opt["accuracy"],
        "test_mcc_opt":      test_opt["mcc"],
        "test_f1_opt":       test_opt["f1"],
        "test_tp":           test_opt["tp"],
        "test_tn":           test_opt["tn"],
        "test_fp":           test_opt["fp"],
        "test_fn":           test_opt["fn"],
        "baseline_test_acc": 0.4914,
        "baseline_test_mcc": -0.0127,
    }
    (args.checkpoint_dir / "metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    print(f"\nSaved final metrics → {args.checkpoint_dir / 'metrics.json'}", flush=True)
    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    main()
