"""Train the paper's full 17-dimensional "Price+TA+Text" model (§4.6, §4.8).

Parallel to ``train.py`` (the canonical 6-feature "Price+Text" pipeline),
which this script never modifies or imports destructively -- it only
imports ``train.py``'s loss/metric/checkpoint helpers, which are already
generic in ``price_dim`` and unaffected by which dataset/config is used.

Writes to separate default checkpoint/TensorBoard directories
(``checkpoints/cmin-us-17feat``, ``runs/cmin-us-17feat``) so a 17-feature
run can never collide with, or overwrite, a 6-feature run's artifacts.

The frozen mT5 text-embedding cache built by ``prepare_embeddings.py`` is
reused unchanged -- text embeddings do not depend on price-feature width,
so there is no separate "17-feature" embedding-preparation step.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config_17feat import Config17Feature
from src.data_17feat import CMINWindowDataset17Feat
from src.model import HierarchicalCoAttentionStockPredictor
from train import Tee, checkpoint, evaluate, metrics, validate_checkpoint_compatibility


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the co-attention model on CMIN 17-dim Price+TA+Text data."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/CMIN-Dataset-official/CMIN-US"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/cmin-us-mt5"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/cmin-us-17feat"))
    parser.add_argument("--epochs", type=int, default=Config17Feature.epochs)
    parser.add_argument(
        "--patience",
        type=int,
        default=Config17Feature.patience,
        help="stop after N epochs without validation-MCC improvement (0 disables early stopping)",
    )
    parser.add_argument("--batch-size", type=int, default=Config17Feature.batch_size)
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
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=Path("runs/cmin-us-17feat"),
        help="Directory to save TensorBoard logs (set to empty string to disable)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save a periodic checkpoint every N epochs inside --checkpoint-dir/periodic/ (0 disables).",
    )
    args = parser.parse_args()

    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = Tee(sys.stdout, args.log_file)  # type: ignore[assignment]

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = Config17Feature()
    config.epochs = args.epochs
    config.patience = args.patience

    train_set = CMINWindowDataset17Feat(args.dataset_root, args.cache_root, "train", seq_len=config.seq_len, max_stocks=args.max_stocks)
    val_set   = CMINWindowDataset17Feat(args.dataset_root, args.cache_root, "val",   seq_len=config.seq_len, max_stocks=args.max_stocks)
    test_set  = CMINWindowDataset17Feat(args.dataset_root, args.cache_root, "test",  seq_len=config.seq_len, max_stocks=args.max_stocks)

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
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size)
    optimizer    = Adam(model.parameters(), lr=config.lr)
    # Table 2: decay rate 1e-4 every five epochs.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_decay_epoch, gamma=1 - config.lr_decay
    )
    criterion    = nn.BCEWithLogitsLoss()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.tensorboard_dir and str(args.tensorboard_dir) != "":
        args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
    else:
        tb_writer = None

    periodic_dir = args.checkpoint_dir / "periodic"
    if args.checkpoint_interval > 0:
        periodic_dir.mkdir(parents=True, exist_ok=True)

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
            validate_checkpoint_compatibility(
                saved,
                price_dim=train_set.price_dim,
                text_embedding_dim=train_set.text_embedding_dim,
                config=config,
            )
            model.load_state_dict(saved["model_state_dict"])
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            history      = saved.get("history", [])
            start_epoch  = saved["epoch"] + 1
            if history:
                best_mcc      = max(r["val_mcc"]      for r in history)
                best_accuracy = max(r["val_accuracy"] for r in history)
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
        for prices, text, labels, _next_return in train_loader:
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
            "train_f1": train_metrics["f1"],
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_mcc": validation["mcc"],
            "val_f1": validation["f1"],
        }
        history.append(result)

        if tb_writer:
            tb_writer.add_scalar("Loss/train",       result["train_loss"],     epoch)
            tb_writer.add_scalar("Loss/val",          result["val_loss"],       epoch)
            tb_writer.add_scalar("Accuracy/train",    result["train_accuracy"], epoch)
            tb_writer.add_scalar("Accuracy/val",      result["val_accuracy"],   epoch)
            tb_writer.add_scalar("MCC/train",         result["train_mcc"],      epoch)
            tb_writer.add_scalar("MCC/val",           result["val_mcc"],        epoch)
            tb_writer.add_scalar("F1/train",          result["train_f1"],       epoch)
            tb_writer.add_scalar("F1/val",            result["val_f1"],         epoch)
            tb_writer.add_scalar("Learning_Rate",     result["learning_rate"],  epoch)
            tb_writer.add_scalar("Epoch_Time_s",      result["elapsed_seconds"],epoch)
            tb_writer.flush()

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
            csv_writer = csv.DictWriter(file, fieldnames=list(result))
            csv_writer.writeheader(); csv_writer.writerows(history)
        tags = []
        if is_best_mcc:
            tags.append("saved best.pt (MCC)")
        if is_best_accuracy:
            tags.append("saved best_accuracy.pt")
        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            periodic_path = periodic_dir / f"epoch_{epoch:03d}.pt"
            checkpoint(
                periodic_path, model=model, optimizer=optimizer, epoch=epoch,
                config=config, result=result, history=history,
            )
            tags.append(f"saved periodic/epoch_{epoch:03d}.pt")
        print(
            f"Epoch {epoch:03d}/{args.epochs} | {result['elapsed_seconds']:.1f}s | "
            f"train loss {result['train_loss']:.4f}, acc {result['train_accuracy']:.4f}, MCC {result['train_mcc']:.4f}, F1 {result['train_f1']:.4f} | "
            f"val loss {result['val_loss']:.4f}, acc {result['val_accuracy']:.4f}, MCC {result['val_mcc']:.4f}, F1 {result['val_f1']:.4f} | "
            f"lr {result['learning_rate']:.2e}" + (" | " + ", ".join(tags) if tags else ""),
            flush=True,
        )
        scheduler.step()
        if config.patience > 0 and stale_epochs >= config.patience:
            print(f"Early stopping after {config.patience} epochs without validation-MCC improvement.", flush=True)
            break

    # The test split is touched exactly once, after validation-based model selection.
    best_path = args.checkpoint_dir / "best.pt"
    if best_path.exists():
        selected = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(selected["model_state_dict"])
    test_result = evaluate(model, test_loader, device, criterion)
    final_result = {**history[-1], "test_loss": test_result["loss"],
                    "test_accuracy": test_result["accuracy"], "test_mcc": test_result["mcc"],
                    "test_f1": test_result["f1"]}
    print(
        f"Held-out test | loss {test_result['loss']:.4f}, "
        f"acc {test_result['accuracy']:.4f}, MCC {test_result['mcc']:.4f}, F1 {test_result['f1']:.4f}",
        flush=True,
    )
    (args.checkpoint_dir / "metrics.json").write_text(json.dumps(final_result, indent=2) + "\n")
    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    main()
