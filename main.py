import argparse
import sys

sys.path.insert(0, ".")

from config import Config
from src.train import run_training


def main():
    parser = argparse.ArgumentParser(description="Stock trend prediction model from paper")
    parser.add_argument("--dataset", type=str, default="CMIN-US", choices=["CMIN-US", "CMIN-CN"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-fusion-layers", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.dataset = args.dataset
    cfg.data_dir = args.data_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.device is not None:
        cfg.device = args.device
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.n_heads is not None:
        cfg.n_heads = args.n_heads
    if args.n_fusion_layers is not None:
        cfg.n_fusion_layers = args.n_fusion_layers

    # Paper defaults for CMIN-CN (Table 8)
    if cfg.dataset == "CMIN-CN":
        cfg.n_heads = 4
        cfg.n_fusion_layers = 4

    print(f"Dataset: {cfg.dataset}")
    print(f"Config: seq_len={cfg.seq_len}, patch_len={cfg.patch_len}, stride={cfg.stride}")
    print(f"         d_model={cfg.d_model}, n_heads={cfg.n_heads}, n_layers={cfg.n_fusion_layers}")
    print(f"         lr={cfg.lr}, epochs={cfg.epochs}, batch_size={cfg.batch_size}")

    run_training(cfg)


if __name__ == "__main__":
    main()
