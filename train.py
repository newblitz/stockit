"""Train the paper model from cached CMIN text embeddings and save checkpoints.

Pass --resume to continue from an existing last.pt checkpoint.  The script
automatically restores model weights, optimizer state, epoch counter, metric
history, and best-metric thresholds so training carries on exactly where it
left off.
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
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import Config
from src.data import CMINWindowDataset
from src.model import HierarchicalCoAttentionStockPredictor


def metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    prediction, truth = (logits.sigmoid() >= 0.5).long().flatten(), labels.long().flatten()
    tp = ((prediction == 1) & (truth == 1)).sum().item()
    tn = ((prediction == 0) & (truth == 0)).sum().item()
    fp = ((prediction == 1) & (truth == 0)).sum().item()
    fn = ((prediction == 0) & (truth == 1)).sum().item()
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator**0.5
    return {"accuracy": (tp + tn) / max(tp + tn + fp + fn, 1), "mcc": mcc}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> dict[str, float]:
    model.eval()
    outputs, labels, losses = [], [], []
    for prices, text, label in loader:
        logits = model(prices.to(device), text.to(device))
        losses.append(criterion(logits, label.to(device)).item())
        outputs.append(logits.cpu())
        labels.append(label)
    result = metrics(torch.cat(outputs), torch.cat(labels))
    result["loss"] = float(np.mean(losses))
    return result


def checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Adam,
    epoch: int,
    config: Config,
    result: dict[str, float],
    history: list[dict[str, float]],
) -> None:
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
         "config": asdict(config), "validation": result, "history": history},
        path,
    )


class Tee:
    """Mirror epoch output to the terminal and an on-disk log."""

    def __init__(self, stream: object, log_file: Path) -> None:
        self.stream = stream
        self.file = log_file.open("a", encoding="utf-8", buffering=1)

    def write(self, message: str) -> int:
        self.stream.write(message)  # type: ignore[attr-defined]
        return self.file.write(message)

    def flush(self) -> None:
        self.stream.flush()  # type: ignore[attr-defined]
        self.file.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the co-attention model on CMIN text+price data."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/CMIN-Dataset-official/CMIN-US"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/cmin-us-mt5"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/cmin-us"))
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument(
        "--patience",
        type=int,
        default=Config.patience,
        help="stop after N epochs without validation-MCC improvement (0 disables early stopping)",
    )
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--max-stocks", type=int, help="use a small subset for a smoke run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", type=Path, help="Mirror epoch metrics to this file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume training from last.pt in --checkpoint-dir. "
            "Restores model weights, optimizer state, epoch counter, metric history, "
            "and best-MCC/accuracy thresholds so training continues seamlessly."
        ),
    )
    args = parser.parse_args()

    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = Tee(sys.stdout, args.log_file)  # type: ignore[assignment]

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    config = Config()
    config.epochs = args.epochs
    config.patience = args.patience

    train_set = CMINWindowDataset(args.dataset_root, args.cache_root, "train", seq_len=config.seq_len, max_stocks=args.max_stocks)
    val_set   = CMINWindowDataset(args.dataset_root, args.cache_root, "val",   seq_len=config.seq_len, max_stocks=args.max_stocks)

    model = HierarchicalCoAttentionStockPredictor(
        text_embedding_dim=train_set.text_embedding_dim, seq_len=config.seq_len, patch_len=config.patch_len,
        stride=config.stride, price_dim=train_set.price_dim, d_model=config.d_model, d_ff=config.d_ff,
        n_heads=config.n_heads, n_layers=config.n_layers, n_fusion_layers=config.n_fusion_layers,
        n_classes=config.n_classes, dropout=config.dropout,
    )
    device = torch.device(args.device)
    model.to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size)
    optimizer    = Adam(model.parameters(), lr=config.lr)
    criterion    = nn.BCEWithLogitsLoss()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume from last.pt if requested ────────────────────────────────────
    start_epoch   = 1
    best_mcc      = float("-inf")
    best_accuracy = float("-inf")
    stale_epochs  = 0
    history: list[dict[str, float]] = []

    last_ckpt = args.checkpoint_dir / "last.pt"
    if args.resume:
        if not last_ckpt.exists():
            print(
                f"[resume] WARNING: --resume was set but {last_ckpt} does not exist. "
                "Starting from scratch.",
                flush=True,
            )
        else:
            print(f"[resume] Loading checkpoint from {last_ckpt} ...", flush=True)
            saved = torch.load(last_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(saved["model_state_dict"])
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            history      = saved.get("history", [])
            start_epoch  = saved["epoch"] + 1          # next epoch to run
            # Restore best-metric thresholds from history so saved best.pt is respected
            if history:
                best_mcc      = max(r["val_mcc"]      for r in history)
                best_accuracy = max(r["val_accuracy"] for r in history)
                # Count stale epochs since the last MCC improvement
                for r in reversed(history):
                    if r["val_mcc"] >= best_mcc:
                        break
                    stale_epochs += 1
            print(
                f"[resume] Resumed from epoch {saved['epoch']} | "
                f"best val MCC so far: {best_mcc:.4f} | "
                f"will train epochs {start_epoch}–{args.epochs}",
                flush=True,
            )

    if start_epoch > args.epochs:
        print(
            f"[resume] Already completed {saved['epoch']} epochs (target: {args.epochs}). "
            "Nothing to do. Increase --epochs to train further.",
            flush=True,
        )
        return

    print(
        f"Training on {device}: train={len(train_set):,}, validation={len(val_set):,}, "
        f"price_dim={train_set.price_dim}, text_dim={train_set.text_embedding_dim}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses, outputs, labels_all = [], [], []
        for prices, text, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(prices.to(device), text.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            outputs.append(logits.detach().cpu())
            labels_all.append(labels)
        train_metrics = metrics(torch.cat(outputs), torch.cat(labels_all))
        validation = evaluate(model, val_loader, device, criterion)
        result = {
            "epoch": epoch,
            "elapsed_seconds": round(time.perf_counter() - epoch_started, 2),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": float(np.mean(losses)),
            "train_accuracy": train_metrics["accuracy"],
            "train_mcc": train_metrics["mcc"],
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_mcc": validation["mcc"],
        }
        history.append(result)
        checkpoint(
            last_ckpt, model=model, optimizer=optimizer, epoch=epoch,
            config=config, result=result, history=history,
        )
        is_best_mcc      = result["val_mcc"]      > best_mcc
        is_best_accuracy = result["val_accuracy"] > best_accuracy
        if is_best_mcc:
            best_mcc, stale_epochs = result["val_mcc"], 0
            checkpoint(
                args.checkpoint_dir / "best.pt", model=model, optimizer=optimizer, epoch=epoch,
                config=config, result=result, history=history,
            )
        else:
            stale_epochs += 1
        if is_best_accuracy:
            best_accuracy = result["val_accuracy"]
            checkpoint(
                args.checkpoint_dir / "best_accuracy.pt", model=model, optimizer=optimizer, epoch=epoch,
                config=config, result=result, history=history,
            )
        (args.checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        with (args.checkpoint_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(result))
            writer.writeheader(); writer.writerows(history)
        tags = []
        if is_best_mcc:
            tags.append("saved best.pt (MCC)")
        if is_best_accuracy:
            tags.append("saved best_accuracy.pt")
        print(
            f"Epoch {epoch:03d}/{args.epochs} | {result['elapsed_seconds']:.1f}s | "
            f"train loss {result['train_loss']:.4f}, acc {result['train_accuracy']:.4f}, MCC {result['train_mcc']:.4f} | "
            f"val loss {result['val_loss']:.4f}, acc {result['val_accuracy']:.4f}, MCC {result['val_mcc']:.4f} | "
            f"lr {result['learning_rate']:.2e}" + (" | " + ", ".join(tags) if tags else ""),
            flush=True,
        )
        if config.patience > 0 and stale_epochs >= config.patience:
            print(f"Early stopping after {config.patience} epochs without validation-MCC improvement.", flush=True)
            break

    (args.checkpoint_dir / "metrics.json").write_text(json.dumps(history[-1], indent=2) + "\n")


if __name__ == "__main__":
    main()
