"""
Simple evaluation script demonstrating how to compute Morph-Net metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluate_pannuke import evaluate_checkpoint

DEMO_CHECKPOINT = Path(
    "output/pannuke_demo-split1-densenet-area-major_axis_length-minor_axis_length-perimeter-eccentricity-scaled.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo evaluation for a Morph-Net checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEMO_CHECKPOINT, help="Path to a `.pt` checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/PanNuke"), help="Root PanNuke directory.")
    parser.add_argument("--split-no", type=int, default=None, help="Override stored split (defaults to checkpoint metadata).")
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        choices=["train", "tr", "val", "validation", "test", "ts"],
        help="Which subset of the split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-scaled", action="store_true", help="Keep metrics in the normalized target space.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    summary, per_class, metadata, _ = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split_no=args.split_no,
        subset=args.subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_scaled=args.target_scaled,
        per_class=True,
        device=device,
    )

    print(f"Evaluation on split {metadata['split_no']} ({metadata['subset']}) for {metadata['checkpoint']}")
    for feat, metrics in summary.items():
        print(
            f"{feat:25s} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f} "
            f"Spearman={metrics['spearman']:.4f} Pearson={metrics['pearson']:.4f}"
        )

    if per_class:
        print("\nPer-class metrics (0: Neoplastic, 1: Inflammatory, 2: Connective, 3: Epithelial)")
        for feat, class_dict in per_class.items():
            for cid, metrics in class_dict.items():
                print(
                    f"{feat:25s} class={cid} RMSE={metrics['rmse']:.4f} "
                    f"R2={metrics['r2']:.4f} Spearman={metrics['spearman']:.4f} "
                    f"Pearson={metrics['pearson']:.4f}"
                )


if __name__ == "__main__":
    main()
