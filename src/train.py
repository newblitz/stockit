import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CMINDataset
from .model import StockModel


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute ACC and MCC (Eq 18-19)."""
    preds = (y_pred >= 0.5).astype(float)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)

    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = (tp * tn - fp * fn) / denom

    return {"acc": acc, "mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        price = batch["price"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        pred = model(ids, mask, price)
        loss = criterion(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(label)
        n += len(label)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n = 0
    criterion = torch.nn.BCELoss()

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        price = batch["price"].to(device)
        label = batch["label"].to(device)

        pred = model(ids, mask, price)
        loss = criterion(pred, label)
        total_loss += loss.item() * len(label)
        n += len(label)

        all_preds.append(pred.cpu().numpy())
        all_labels.append(label.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = compute_metrics(labels, preds)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def run_training(cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    common = dict(
        data_dir=cfg.data_dir,
        dataset=cfg.dataset,
        tokenizer_name=cfg.llm_model,
        seq_len=cfg.seq_len,
        max_text_len=cfg.max_text_len,
    )
    train_ds = CMINDataset(**common, split="train")
    val_ds = CMINDataset(**common, split="val")
    test_ds = CMINDataset(**common, split="test")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = StockModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=1e-5
    )
    criterion = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_epoch, gamma=1 - cfg.lr_decay)

    # Training loop
    best_val_mcc = -1
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:2d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val ACC: {val_metrics['acc']*100:.2f}% | "
            f"Val MCC: {val_metrics['mcc']:.4f}"
        )

        if val_metrics["mcc"] > best_val_mcc:
            best_val_mcc = val_metrics["mcc"]
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest ACC: {test_metrics['acc']*100:.2f}% | Test MCC: {test_metrics['mcc']:.4f}")
    print(f"TP={test_metrics['tp']} TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']}")
    return test_metrics
