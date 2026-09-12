"""Compatibility entry point for the paper-faithful training pipeline.

Use ``python main.py --help`` or ``python train.py --help``; both now invoke
the same cached-embedding, hierarchical-summary architecture.
"""

from train import main


if __name__ == "__main__":
    main()
