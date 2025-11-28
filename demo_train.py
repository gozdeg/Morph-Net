"""
Quick-start training script that mirrors the hyper-parameters used in the
original Morph-Net training notebook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from train_pannuke import DEFAULT_FEAT_TYPES, TrainConfig, train


def parse_args():
    parser = argparse.ArgumentParser(description="Demo Morph-Net training run.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/PanNuke"), help="Root PanNuke directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Where checkpoints will be stored.")
    parser.add_argument("--device-id", type=str, default="0", help="CUDA device id (as string).")
    parser.add_argument("--split-no", type=int, default=1, choices=[1, 2, 3], help="PanNuke split to train on.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs (matches the original training notebook; reduce for quick tests).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device_id=args.device_id,
        split_no=args.split_no,
        base_model="densenet",
        feat_types=list(DEFAULT_FEAT_TYPES),
        model_name_prefix="pannuke_demo",
        batch_size=128,
        num_workers=4,
        num_epochs=args.epochs,
        learning_rate=1e-3,
        awl_learning_rate=1e-3,
        pretrained=True,
        use_lr_scheduler=True,
        scheduler_step_size=25,
        validation_ratio=1.0,
        early_stop_patience=100,
        scaled_targets=True,
        use_awl=True,
        freeze_base_epochs=0,
        lr_decay_on_unfreeze=0.1,
    )
    print("Running demo training with config:")
    for k, v in config.as_dict().items():
        print(f"  {k}: {v}")
    train(config)


if __name__ == "__main__":
    main()
