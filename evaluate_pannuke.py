from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from dataset_pannuke import PANNNUKE_SPLITS, get_postprep_data
from dataset_utils import CustomDataset, get_transforms, prep_features_target
from test_utils import load_checkpoint, results, results_classwise, run_inference

SUBSET_ALIASES = {
    "train": "tr",
    "tr": "tr",
    "val": "val",
    "validation": "val",
    "test": "ts",
    "ts": "ts",
}


def build_loader(fold_id, data_dir, feat_names, batch_size, num_workers, target_scaled):
    fold = get_postprep_data(fold_id, feat_names=feat_names, data_dir=data_dir)
    targets, _, _ = prep_features_target(fold["features"], scaled=target_scaled, feat_keys=feat_names)
    dataset = CustomDataset(fold["imgs"], targets, transform=get_transforms("val"))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    data_dir: Path | str | None,
    split_no: int | None = None,
    subset: str = "test",
    batch_size: int = 64,
    num_workers: int = 4,
    target_scaled: bool = False,
    per_class: bool = False,
    device: torch.device | None = None,
) -> Tuple[Dict, Dict | None, Dict, Dict]:
    """
    Programmatic helper to evaluate a Morph-Net checkpoint.
    Returns (summary_metrics, per_class_metrics, metadata, checkpoint_dict).
    """
    checkpoint_path = Path(checkpoint_path)
    subset_key = SUBSET_ALIASES.get(subset, subset)
    if subset_key not in {"tr", "val", "ts"}:
        raise ValueError(f"Unsupported subset: {subset}")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})
    feat_names = checkpoint.get("feature_names")
    if feat_names is None:
        raise ValueError("Checkpoint is missing feature_names. Re-train with the updated training script to evaluate.")

    if data_dir is None:
        stored_dir = config.get("data_dir")
        if stored_dir is None:
            raise ValueError("data_dir is required because the checkpoint does not include the original path.")
        data_dir = stored_dir
    data_dir = Path(data_dir)

    split_no = split_no or config.get("split_no")
    if split_no not in PANNNUKE_SPLITS:
        raise ValueError(f"Unknown split number {split_no}.")

    fold_id = PANNNUKE_SPLITS[split_no][subset_key]
    loader = build_loader(fold_id, data_dir, feat_names, batch_size, num_workers, target_scaled)

    preds, targets = run_inference(model, loader, device)
    should_rescale = bool(config.get("scaled_targets", True) and not target_scaled)
    summary = results(
        preds,
        targets,
        feat_names,
        tr_feature_stats=checkpoint.get("feature_stats"),
        rescale=should_rescale,
        target_scaled=target_scaled,
    )

    per_class_summary = None
    if per_class:
        per_class_summary = results_classwise(
            preds,
            targets,
            feat_names,
            tr_feature_stats=checkpoint.get("feature_stats"),
            rescale=should_rescale,
            target_scaled=target_scaled,
        )

    metadata = {
        "split_no": split_no,
        "subset": subset,
        "fold_id": fold_id,
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
    }
    return summary, per_class_summary, metadata, checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Morph-Net checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the .pt file produced by train_pannuke.py")
    parser.add_argument("--data-dir", type=Path, help="Root PanNuke directory. Defaults to the value stored in the checkpoint config.")
    parser.add_argument("--split-no", type=int, help="Override split number stored in the checkpoint.")
    parser.add_argument("--subset", choices=list(SUBSET_ALIASES.keys()), default="test", help="Which fold to evaluate.")
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-scaled", action="store_true", help="Keep evaluation targets scaled to [0,1].")
    parser.add_argument("--per-class", action="store_true", help="Also report per-class metrics.")
    parser.add_argument("--save-json", type=Path, help="Optional path to store the metrics as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    summary, class_summary, metadata, _ = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split_no=args.split_no,
        subset=args.subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_scaled=args.target_scaled,
        per_class=args.per_class,
        device=device,
    )

    print(f"Evaluation on split {metadata['split_no']} ({metadata['subset']})")
    for feat, metrics_dict in summary.items():
        print(
            f"{feat:25s} RMSE={metrics_dict['rmse']:.4f} "
            f"R2={metrics_dict['r2']:.4f} "
            f"Spearman={metrics_dict['spearman']:.4f} "
            f"Pearson={metrics_dict['pearson']:.4f}"
        )

    if class_summary:
        print()
        print("Per-class metrics (0: Neoplastic, 1: Inflammatory, 2: Connective, 3: Epithelial)")
        for feat, class_dict in class_summary.items():
            for class_id, metrics_dict in class_dict.items():
                print(
                    f"{feat:25s} class={class_id} "
                    f"RMSE={metrics_dict['rmse']:.4f} "
                    f"R2={metrics_dict['r2']:.4f} "
                    f"Spearman={metrics_dict['spearman']:.4f} "
                    f"Pearson={metrics_dict['pearson']:.4f}"
                )

    if args.save_json:
        payload = {
            "split_no": metadata["split_no"],
            "subset": metadata["subset"],
            "metrics": summary,
            "per_class": class_summary,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to {args.save_json}")


if __name__ == "__main__":
    main()
