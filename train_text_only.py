"""
Train the CMIN-US Text-Only ablation.

Uses:
    - ONLY daily mT5 text embeddings
    - NO price features
    - NO price-text alignment
    - NO co-attention
    - Patch-based Transformer encoder
    - Binary next-day direction prediction

Paper reference:
    Text only
    CMIN-US:
        ACC = 56.65 +/- 0.30 %
        MCC = 0.125 +/- 0.008
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
from torch.utils.tensorboard import SummaryWriter

from config import Config
from src.data import CMINWindowDataset
from src.model import SingleChannelPatchTransformer


# ============================================================
# METRICS
# ============================================================

def metrics(
    logits: Tensor,
    labels: Tensor,
) -> dict[str, float]:

    prediction = (
        logits.sigmoid() >= 0.5
    ).long().flatten()

    truth = labels.long().flatten()

    tp = (
        (prediction == 1)
        & (truth == 1)
    ).sum().item()

    tn = (
        (prediction == 0)
        & (truth == 0)
    ).sum().item()

    fp = (
        (prediction == 1)
        & (truth == 0)
    ).sum().item()

    fn = (
        (prediction == 0)
        & (truth == 1)
    ).sum().item()

    denominator = float(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    mcc = (
        0.0
        if denominator == 0
        else (tp * tn - fp * fn)
        / denominator ** 0.5
    )

    precision = (
        tp / max(tp + fp, 1)
    )

    recall = (
        tp / max(tp + fn, 1)
    )

    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall
        / (precision + recall)
    )

    accuracy = (
        (tp + tn)
        / max(tp + tn + fp + fn, 1)
    )

    return {
        "accuracy": accuracy,
        "mcc": mcc,
        "f1": f1,
    }


# ============================================================
# TEXT-ONLY MODEL
# ============================================================

class TextOnlyStockPredictor(nn.Module):
    """
    Text-only version of the CMIN model.

    Input:
        text_embeddings:
            [B, 30, 768]

    Processing:
        768 -> 6 projection
        30-day sequence
        patch Transformer
        classification

    There is NO:
        - price input
        - price encoder
        - price-text alignment
        - co-attention
    """

    def __init__(
        self,
        *,
        text_embedding_dim: int,
        seq_len: int = 30,
        patch_len: int = 10,
        stride: int = 5,
        model_dim: int = 6,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 16,
        n_layers: int = 4,
        n_classes: int = 1,
        dropout: float = 0.2,
    ) -> None:

        super().__init__()

        self.text_embedding_dim = (
            text_embedding_dim
        )

        self.model_dim = model_dim

        # Project mT5 embedding into the feature
        # dimension used by the patch Transformer.
        #
        # 768 -> 6
        self.text_projection = nn.Sequential(
            nn.Linear(
                text_embedding_dim,
                model_dim,
            ),
            nn.Dropout(dropout),
        )

        # Same patch-based Transformer used in
        # your existing model.
        self.text_encoder = (
            SingleChannelPatchTransformer(
                seq_len=seq_len,
                patch_len=patch_len,
                stride=stride,
                price_dim=model_dim,
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
            )
        )

        n_patches = (
            self.text_encoder.num_patches
        )

        # model_dim × patches × hidden
        self.classifier = nn.Linear(
            model_dim
            * n_patches
            * d_model,
            n_classes,
        )

    def forward(
        self,
        text_embeddings: Tensor,
    ) -> Tensor:

        if text_embeddings.ndim != 3:

            raise ValueError(
                "Expected text embeddings with "
                "shape [B, L, D]"
            )

        batch, length, dim = (
            text_embeddings.shape
        )

        if dim != self.text_embedding_dim:

            raise ValueError(
                f"Expected text dimension "
                f"{self.text_embedding_dim}, "
                f"got {dim}"
            )

        # ----------------------------------------------------
        # 768 -> 6
        # ----------------------------------------------------

        text = self.text_projection(
            text_embeddings
        )

        # text:
        # [B, 30, 6]

        # ----------------------------------------------------
        # PATCH TRANSFORMER
        # ----------------------------------------------------

        encoded = self.text_encoder(
            text
        )

        # encoded:
        # [B * 6, N, 128]

        # ----------------------------------------------------
        # RESTORE BATCH DIMENSION
        # ----------------------------------------------------

        features = encoded.reshape(
            batch,
            self.model_dim,
            encoded.size(1),
            encoded.size(2),
        )

        # [B, 6, N, 128]

        features = features.flatten(1)

        # [B, 6*N*128]

        logits = self.classifier(
            features
        )

        return logits


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:

    model.eval()

    outputs = []
    labels = []
    losses = []

    n_examples = 0

    for (
        _prices,
        text,
        label,
        _next_return,
    ) in loader:

        text = text.to(device)
        label = label.to(device)

        logits = model(text)

        loss = criterion(
            logits,
            label,
        )

        losses.append(
            loss.item() * len(label)
        )

        n_examples += len(label)

        outputs.append(
            logits.cpu()
        )

        labels.append(
            label.cpu()
        )

    all_outputs = torch.cat(
        outputs
    )

    all_labels = torch.cat(
        labels
    )

    result = metrics(
        all_outputs,
        all_labels,
    )

    result["loss"] = (
        float(sum(losses))
        / max(n_examples, 1)
    )

    return result


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
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
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "config":
                asdict(config),
            "validation":
                result,
            "history":
                history,
            "experiment":
                "text_only",
        },
        path,
    )


# ============================================================
# LOGGING
# ============================================================

class Tee:

    def __init__(
        self,
        stream: object,
        log_file: Path,
    ) -> None:

        self.stream = stream

        self.file = log_file.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )

    def write(
        self,
        message: str,
    ) -> int:

        self.stream.write(
            message
        )  # type: ignore[attr-defined]

        return self.file.write(
            message
        )

    def flush(self) -> None:

        self.stream.flush()
        self.file.flush()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train CMIN-US Text-Only "
            "ablation."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "data/CMIN-Dataset-official/"
            "CMIN-US"
        ),
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            "data/cache/cmin-us-mt5"
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "checkpoints/"
            "cmin-us-text-only"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=Config.epochs,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=Config.patience,
        help=(
            "Early stopping patience. "
            "Use 0 to disable."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.batch_size,
    )

    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=Path(
            "runs/cmin-us-text-only"
        ),
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    # ========================================================
    # SEED
    # ========================================================

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ========================================================
    # LOG FILE
    # ========================================================

    if args.log_file:

        args.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sys.stdout = Tee(
            sys.stdout,
            args.log_file,
        )  # type: ignore[assignment]

    # ========================================================
    # CONFIG
    # ========================================================

    config = Config()

    config.epochs = args.epochs
    config.patience = args.patience

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print("=" * 70)
    print("TEXT-ONLY CMIN-US EXPERIMENT")
    print("=" * 70)

    print(
        f"Dataset  : {args.dataset_root}"
    )

    print(
        f"Cache    : {args.cache_root}"
    )

    print(
        f"Checkpts : {args.checkpoint_dir}"
    )

    print(
        f"Device   : {args.device}"
    )

    print()

    # Same dataset construction as your
    # Price+Text experiment.
    #
    # We use the text embeddings from the
    # fourth element of the dataset tuple.
    train_set = CMINWindowDataset(
        args.dataset_root,
        args.cache_root,
        "train",
        seq_len=config.seq_len,
        max_stocks=args.max_stocks,
    )

    val_set = CMINWindowDataset(
        args.dataset_root,
        args.cache_root,
        "val",
        seq_len=config.seq_len,
        max_stocks=args.max_stocks,
    )

    test_set = CMINWindowDataset(
        args.dataset_root,
        args.cache_root,
        "test",
        seq_len=config.seq_len,
        max_stocks=args.max_stocks,
    )

    # ========================================================
    # TEXT DIMENSION
    # ========================================================

    text_dim = (
        train_set.text_embedding_dim
    )

    if text_dim is None:

        raise ValueError(
            "Could not determine text "
            "embedding dimension."
        )

    print(
        f"Text embedding dimension: "
        f"{text_dim}"
    )

    print(
        f"Train samples: "
        f"{len(train_set):,}"
    )

    print(
        f"Val samples  : "
        f"{len(val_set):,}"
    )

    print(
        f"Test samples : "
        f"{len(test_set):,}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = TextOnlyStockPredictor(
        text_embedding_dim=text_dim,
        seq_len=config.seq_len,
        patch_len=config.patch_len,
        stride=config.stride,
        model_dim=6,
        d_model=config.d_model,
        d_ff=config.d_ff,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        n_classes=config.n_classes,
        dropout=config.dropout,
    )

    device = torch.device(
        args.device
    )

    model.to(device)

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
    )

    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_decay_epoch,
            gamma=1 - config.lr_decay,
        )
    )

    criterion = nn.BCEWithLogitsLoss()

    # ========================================================
    # CHECKPOINT DIRECTORY
    # ========================================================

    args.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    periodic_dir = (
        args.checkpoint_dir
        / "periodic"
    )

    if args.checkpoint_interval > 0:

        periodic_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # TENSORBOARD
    # ========================================================

    if (
        args.tensorboard_dir
        and str(args.tensorboard_dir) != ""
    ):

        args.tensorboard_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        tb_writer = SummaryWriter(
            log_dir=str(
                args.tensorboard_dir
            )
        )

    else:

        tb_writer = None

    # ========================================================
    # MODEL INFO
    # ========================================================

    n_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Trainable parameters: "
        f"{n_parameters:,}"
    )

    print(
        f"Input text dimension : "
        f"{text_dim}"
    )

    print(
        f"Projected dimension  : 6"
    )

    print(
        f"Sequence length      : "
        f"{config.seq_len}"
    )

    print(
        f"Patch length         : "
        f"{config.patch_len}"
    )

    print(
        f"Patch stride         : "
        f"{config.stride}"
    )

    print(
        f"d_model              : "
        f"{config.d_model}"
    )

    print(
        f"d_ff                 : "
        f"{config.d_ff}"
    )

    print(
        f"Attention heads      : "
        f"{config.n_heads}"
    )

    print(
        f"Transformer layers   : "
        f"{config.n_layers}"
    )

    print(
        f"Dropout              : "
        f"{config.dropout}"
    )

    print(
        f"Learning rate        : "
        f"{config.lr}"
    )

    print(
        f"Batch size           : "
        f"{args.batch_size}"
    )

    print(
        f"Epochs               : "
        f"{args.epochs}"
    )

    print(
        f"Patience             : "
        f"{args.patience}"
    )

    print()

    # ========================================================
    # TRAIN
    # ========================================================

    best_mcc = float("-inf")
    best_accuracy = float("-inf")
    stale_epochs = 0

    history = []

    print("=" * 70)
    print("STARTING TEXT-ONLY TRAINING")
    print("=" * 70)
    print()

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_started = (
            time.perf_counter()
        )

        model.train()

        losses = []
        outputs = []
        labels_all = []

        # ----------------------------------------------------
        # TRAINING BATCHES
        # ----------------------------------------------------

        for (
            _prices,
            text,
            labels,
            _next_return,
        ) in train_loader:

            text = text.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            # TEXT ONLY
            logits = model(text)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            losses.append(
                loss.item()
            )

            outputs.append(
                logits.detach().cpu()
            )

            labels_all.append(
                labels.detach().cpu()
            )

        # ----------------------------------------------------
        # TRAIN METRICS
        # ----------------------------------------------------

        train_metrics = metrics(
            torch.cat(outputs),
            torch.cat(labels_all),
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        elapsed = (
            time.perf_counter()
            - epoch_started
        )

        result = {

            "epoch": epoch,

            "elapsed_seconds":
                round(
                    elapsed,
                    2,
                ),

            "learning_rate":
                optimizer.param_groups[
                    0
                ]["lr"],

            "train_loss":
                float(
                    np.mean(losses)
                ),

            "train_accuracy":
                train_metrics[
                    "accuracy"
                ],

            "train_mcc":
                train_metrics[
                    "mcc"
                ],

            "train_f1":
                train_metrics[
                    "f1"
                ],

            "val_loss":
                validation[
                    "loss"
                ],

            "val_accuracy":
                validation[
                    "accuracy"
                ],

            "val_mcc":
                validation[
                    "mcc"
                ],

            "val_f1":
                validation[
                    "f1"
                ],
        }

        history.append(result)

        # ====================================================
        # TENSORBOARD
        # ====================================================

        if tb_writer:

            tb_writer.add_scalar(
                "Loss/train",
                result[
                    "train_loss"
                ],
                epoch,
            )

            tb_writer.add_scalar(
                "Loss/val",
                result[
                    "val_loss"
                ],
                epoch,
            )

            tb_writer.add_scalar(
                "Accuracy/train",
                result[
                    "train_accuracy"
                ],
                epoch,
            )

            tb_writer.add_scalar(
                "Accuracy/val",
                result[
                    "val_accuracy"
                ],
                epoch,
            )

            tb_writer.add_scalar(
                "MCC/train",
                result[
                    "train_mcc"
                ],
                epoch,
            )

            tb_writer.add_scalar(
                "MCC/val",
                result[
                    "val_mcc"
                ],
                epoch,
            )

            tb_writer.add_scalar(
                "F1/train",
                result[
                    "train_f1"
                ],
                epoch,
            )

            tb_writer.add_scalar(
                "F1/val",
                result[
                    "val_f1"
                ],
                epoch,
            )

            tb_writer.flush()

        # ====================================================
        # LAST CHECKPOINT
        # ====================================================

        save_checkpoint(
            args.checkpoint_dir
            / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            result=result,
            history=history,
        )

        # ====================================================
        # BEST MCC
        # ====================================================

        is_best_mcc = (
            result["val_mcc"]
            > best_mcc
        )

        is_best_accuracy = (
            result["val_accuracy"]
            > best_accuracy
        )

        tags = []

        if is_best_mcc:

            best_mcc = (
                result["val_mcc"]
            )

            stale_epochs = 0

            save_checkpoint(
                args.checkpoint_dir
                / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                result=result,
                history=history,
            )

            tags.append(
                "saved best.pt (MCC)"
            )

        else:

            stale_epochs += 1

        # ====================================================
        # BEST ACCURACY
        # ====================================================

        if is_best_accuracy:

            best_accuracy = (
                result["val_accuracy"]
            )

            save_checkpoint(
                args.checkpoint_dir
                / "best_accuracy.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                result=result,
                history=history,
            )

            tags.append(
                "saved best_accuracy.pt"
            )

        # ====================================================
        # PERIODIC CHECKPOINT
        # ====================================================

        if (
            args.checkpoint_interval > 0
            and epoch
            % args.checkpoint_interval
            == 0
        ):

            periodic_path = (
                periodic_dir
                / f"epoch_{epoch:03d}.pt"
            )

            save_checkpoint(
                periodic_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                result=result,
                history=history,
            )

            tags.append(
                f"saved periodic/"
                f"epoch_{epoch:03d}.pt"
            )

        # ====================================================
        # SAVE HISTORY
        # ====================================================

        (
            args.checkpoint_dir
            / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            )
            + "\n"
        )

        with (
            args.checkpoint_dir
            / "history.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    result
                ),
            )

            writer.writeheader()
            writer.writerows(
                history
            )

        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{args.epochs} | "
            f"{elapsed:.1f}s | "
            f"train loss "
            f"{result['train_loss']:.4f}, "
            f"acc "
            f"{result['train_accuracy']:.4f}, "
            f"MCC "
            f"{result['train_mcc']:.4f}, "
            f"F1 "
            f"{result['train_f1']:.4f} | "
            f"val loss "
            f"{result['val_loss']:.4f}, "
            f"acc "
            f"{result['val_accuracy']:.4f}, "
            f"MCC "
            f"{result['val_mcc']:.4f}, "
            f"F1 "
            f"{result['val_f1']:.4f} | "
            f"lr "
            f"{result['learning_rate']:.2e}"
            + (
                " | "
                + ", ".join(tags)
                if tags
                else ""
            ),
            flush=True,
        )

        # ====================================================
        # LR
        # ====================================================

        scheduler.step()

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            config.patience > 0
            and stale_epochs
            >= config.patience
        ):

            print(
                f"\nEarly stopping after "
                f"{config.patience} epochs "
                f"without validation-MCC "
                f"improvement.",
                flush=True,
            )

            break

    # ========================================================
    # TEST
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEXT-ONLY TEST")
    print("=" * 70)

    best_path = (
        args.checkpoint_dir
        / "best.pt"
    )

    if best_path.exists():

        selected = torch.load(
            best_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            selected[
                "model_state_dict"
            ]
        )

        print(
            f"Loaded best checkpoint: "
            f"{best_path}"
        )

    test_result = evaluate(
        model,
        test_loader,
        device,
        criterion,
    )

    final_result = {

        **history[-1],

        "test_loss":
            test_result[
                "loss"
            ],

        "test_accuracy":
            test_result[
                "accuracy"
            ],

        "test_mcc":
            test_result[
                "mcc"
            ],

        "test_f1":
            test_result[
                "f1"
            ],
    }

    print()

    print(
        f"Held-out test | "
        f"loss "
        f"{test_result['loss']:.4f}, "
        f"acc "
        f"{test_result['accuracy']:.4f}, "
        f"MCC "
        f"{test_result['mcc']:.4f}, "
        f"F1 "
        f"{test_result['f1']:.4f}",
        flush=True,
    )

    print()

    print(
        "Paper Text-only reference:"
    )

    print(
        "CMIN-US ACC = "
        "56.65 +/- 0.30%"
    )

    print(
        "CMIN-US MCC = "
        "0.125 +/- 0.008"
    )

    # ========================================================
    # SAVE FINAL METRICS
    # ========================================================

    (
        args.checkpoint_dir
        / "metrics.json"
    ).write_text(
        json.dumps(
            final_result,
            indent=2,
        )
        + "\n"
    )

    if tb_writer:
        tb_writer.close()

    print()
    print("=" * 70)
    print(
        "TEXT-ONLY TRAINING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()