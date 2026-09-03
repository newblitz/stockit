"""kaggle_persist.py — Save and restore training artifacts on Kaggle.

Kaggle's /kaggle/working/ directory is wiped when a session ends.
This script uploads checkpoints + embeddings to a Kaggle Dataset so they
survive restarts and can be mounted in a new session.

USAGE (inside a Kaggle notebook):
----------------------------------------------------------------------
# Save (run after training / after embeddings are ready):
!python kaggle_persist.py save \
    --cache-root     /kaggle/working/cache/cmin-us-mt5 \
    --checkpoint-dir /kaggle/working/checkpoints/cmin-us \
    --dataset        YOUR_KAGGLE_USERNAME/stockit-artifacts

# Restore (run at the START of a new session before training):
!python kaggle_persist.py restore \
    --cache-root     /kaggle/working/cache/cmin-us-mt5 \
    --checkpoint-dir /kaggle/working/checkpoints/cmin-us \
    --dataset        YOUR_KAGGLE_USERNAME/stockit-artifacts
----------------------------------------------------------------------

FIRST-TIME SETUP (do this once):
1. Add your Kaggle API key to the notebook secret named KAGGLE_KEY,
   OR set environment variables KAGGLE_USERNAME and KAGGLE_KEY.
2. Create an empty Kaggle Dataset named "stockit-artifacts" on kaggle.com/datasets.
   It can be private; the script will push new versions to it.
----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_kaggle_cli() -> None:
    """Ensure the kaggle CLI is available."""
    result = subprocess.run(["kaggle", "--version"], capture_output=True)
    if result.returncode != 0:
        sys.exit("kaggle CLI not found. Run:  pip install kaggle")


def _setup_kaggle_credentials() -> None:
    """
    Set up ~/.kaggle/kaggle.json from notebook secrets or environment variables.
    In Kaggle notebooks, add secrets named KAGGLE_USERNAME and KAGGLE_KEY.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if kaggle_json.exists():
        return  # already configured

    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")

    # Kaggle notebooks expose secrets via userdata
    if not (username and key):
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore
            secrets  = UserSecretsClient()
            username = username or secrets.get_secret("KAGGLE_USERNAME")
            key      = key      or secrets.get_secret("KAGGLE_KEY")
        except Exception:
            pass

    if not (username and key):
        sys.exit(
            "Kaggle credentials not found.\n"
            "Set notebook secrets KAGGLE_USERNAME and KAGGLE_KEY, "
            "or place kaggle.json at ~/.kaggle/kaggle.json."
        )

    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json.write_text(json.dumps({"username": username, "key": key}))
    kaggle_json.chmod(0o600)
    print(f"Kaggle credentials written to {kaggle_json}", flush=True)


# ── save ──────────────────────────────────────────────────────────────────────

def cmd_save(args: argparse.Namespace) -> None:
    """Zip checkpoints + embeddings and push as a new Kaggle Dataset version."""
    _check_kaggle_cli()
    _setup_kaggle_credentials()

    staging = Path("/kaggle/working/_persist_staging")
    staging.mkdir(parents=True, exist_ok=True)

    # Copy artifacts into staging
    cache_src = Path(args.cache_root)
    ckpt_src  = Path(args.checkpoint_dir)

    if cache_src.exists():
        dest = staging / "cache"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(cache_src, dest)
        pt_count = len(list(dest.rglob("*.pt")))
        print(f"  Staged {pt_count} embedding .pt file(s) from {cache_src}", flush=True)
    else:
        print(f"  WARNING: cache-root {cache_src} does not exist — skipping embeddings.", flush=True)

    if ckpt_src.exists():
        dest = staging / "checkpoints"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(ckpt_src, dest)
        pt_count = len(list(dest.rglob("*.pt")))
        print(f"  Staged {pt_count} checkpoint .pt file(s) from {ckpt_src}", flush=True)
    else:
        print(f"  WARNING: checkpoint-dir {ckpt_src} does not exist — skipping checkpoints.", flush=True)

    # Write dataset-metadata.json so kaggle CLI knows how to create the dataset
    username, dataset_name = args.dataset.split("/", 1)
    metadata = {
        "title": dataset_name,
        "id": args.dataset,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

    # Push new version
    print(f"\nPushing to Kaggle Dataset '{args.dataset}' ...", flush=True)
    result = subprocess.run(
        ["kaggle", "datasets", "version", "-p", str(staging),
         "-m", args.message, "--dir-mode", "zip"],
        capture_output=False,
    )
    if result.returncode == 0:
        print(f"\n✓ Artifacts saved to https://www.kaggle.com/datasets/{args.dataset}", flush=True)
    else:
        # First time: create the dataset instead of versioning it
        print("Dataset may not exist yet — trying to create it ...", flush=True)
        result2 = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(staging), "--dir-mode", "zip"],
            capture_output=False,
        )
        if result2.returncode == 0:
            print(f"\n✓ Dataset created at https://www.kaggle.com/datasets/{args.dataset}", flush=True)
        else:
            sys.exit("Failed to save to Kaggle. Check API credentials and dataset name.")

    shutil.rmtree(staging, ignore_errors=True)


# ── restore ───────────────────────────────────────────────────────────────────

def cmd_restore(args: argparse.Namespace) -> None:
    """Download the Kaggle Dataset and extract checkpoints + embeddings."""
    _check_kaggle_cli()
    _setup_kaggle_credentials()

    staging = Path("/kaggle/working/_persist_restore")
    staging.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{args.dataset}' from Kaggle ...", flush=True)
    result = subprocess.run(
        ["kaggle", "datasets", "download", args.dataset, "-p", str(staging), "--unzip"],
        capture_output=False,
    )
    if result.returncode != 0:
        sys.exit(f"Failed to download dataset '{args.dataset}'.")

    # Restore embeddings
    cache_src = staging / "cache"
    cache_dst = Path(args.cache_root)
    if cache_src.exists():
        cache_dst.mkdir(parents=True, exist_ok=True)
        for pt_file in cache_src.rglob("*.pt"):
            target = cache_dst / pt_file.name
            if not target.exists():
                shutil.copy2(pt_file, target)
        pt_count = len(list(cache_dst.rglob("*.pt")))
        print(f"✓ Restored embeddings → {cache_dst} ({pt_count} .pt files)", flush=True)
    else:
        print("  No embeddings folder found in the saved dataset.", flush=True)

    # Restore checkpoints
    ckpt_src = staging / "checkpoints"
    ckpt_dst = Path(args.checkpoint_dir)
    if ckpt_src.exists():
        ckpt_dst.mkdir(parents=True, exist_ok=True)
        for f in ckpt_src.iterdir():
            target = ckpt_dst / f.name
            shutil.copy2(f, target)
        print(f"✓ Restored checkpoints → {ckpt_dst}", flush=True)
        last = ckpt_dst / "last.pt"
        if last.exists():
            import torch
            saved = torch.load(last, map_location="cpu", weights_only=False)
            print(
                f"  last.pt = epoch {saved['epoch']} | "
                f"val MCC {saved['validation']['val_mcc']:.4f}",
                flush=True,
            )
    else:
        print("  No checkpoints folder found in the saved dataset.", flush=True)

    shutil.rmtree(staging, ignore_errors=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persist / restore Kaggle training artifacts across sessions."
    )
    
    # Common arguments for both save and restore
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dataset",
        required=True,
        metavar="USERNAME/DATASET-NAME",
        help="Kaggle Dataset slug to push to / pull from, e.g. johnsmith/stockit-artifacts",
    )
    common.add_argument(
        "--cache-root",
        default="/kaggle/working/cache/cmin-us-mt5",
        help="Path to the mT5 embedding cache directory.",
    )
    common.add_argument(
        "--checkpoint-dir",
        default="/kaggle/working/checkpoints/cmin-us",
        help="Path to the checkpoint directory.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    save_p = sub.add_parser("save", parents=[common], help="Upload artifacts to a Kaggle Dataset.")
    save_p.add_argument(
        "-m", "--message",
        default="checkpoint update",
        help="Version message for the Kaggle Dataset.",
    )

    sub.add_parser("restore", parents=[common], help="Download artifacts from a Kaggle Dataset.")

    args = parser.parse_args()
    if args.command == "save":
        cmd_save(args)
    elif args.command == "restore":
        cmd_restore(args)


if __name__ == "__main__":
    main()
