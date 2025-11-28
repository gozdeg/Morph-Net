from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from AutomaticWeightedLoss import AutomaticWeightedLoss
from dataset_pannuke import PANNNUKE_SPLITS, get_postprep_data
from dataset_utils import CustomDataset, get_transforms, prep_features_target
from models import custom_model, set_parameter_requires_grad

DEFAULT_FEAT_TYPES = ("area", "major_axis_length", "minor_axis_length", "perimeter", "eccentricity")


@dataclass
class TrainConfig:
    data_dir: Path = Path("data/PanNuke")
    output_dir: Path = Path("output")
    device_id: str = "0"
    split_no: int = 1
    base_model: str = "densenet"
    feat_types: Sequence[str] = field(default_factory=lambda: list(DEFAULT_FEAT_TYPES))
    model_name_prefix: str = "MorphNet"
    nclasses: int = 4
    batch_size: int = 128
    seed: int = 1
    num_workers: int = 4
    learning_rate: float = 1e-3
    awl_learning_rate: float = 1e-3
    pretrained: bool = True
    use_lr_scheduler: bool = True
    scheduler_step_size: int = 25
    num_epochs: int = 1000
    validation_ratio: float = 1.0
    early_stop_patience: int = 100
    scaled_targets: bool = True
    use_awl: bool = False
    freeze_base_epochs: int = 0
    lr_decay_on_unfreeze: float = 0.1

    def __post_init__(self):
        self.data_dir = Path(self.data_dir).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_id = str(self.device_id)
        self.feat_types = list(self.feat_types) if self.feat_types else []

    @property
    def feat_names(self) -> List[str]:
        names = ["count"]
        for feat in self.feat_types:
            names.append(f"total_{feat}")
        return names

    @property
    def num_targets(self) -> int:
        return self.nclasses * len(self.feat_names)

    @property
    def checkpoint_stem(self) -> str:
        feat_suffix = "-".join(self.feat_types) if self.feat_types else "count"
        scale_flag = "scaled" if self.scaled_targets else "raw"
        return f"{self.model_name_prefix}-split{self.split_no}-{self.base_model}-{feat_suffix}-{scale_flag}"

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / f"{self.checkpoint_stem}.pt"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / f"{self.checkpoint_stem}_summary.json"

    def as_dict(self) -> Dict:
        data = {
            **self.__dict__,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "feat_types": list(self.feat_types),
        }
        return data


def set_random_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def shuffle_arrays(x: np.ndarray, y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    return x[idx], y[idx]


def build_dataloaders(config: TrainConfig, fold_ids: Dict[str, int]):
    train_fold = get_postprep_data(fold_ids["tr"], feat_names=config.feat_names, data_dir=config.data_dir)
    val_fold = get_postprep_data(fold_ids["val"], feat_names=config.feat_names, data_dir=config.data_dir)

    tr_targets, _, tr_feature_stats = prep_features_target(
        train_fold["features"], scaled=config.scaled_targets, feat_keys=config.feat_names
    )
    val_targets, _, _ = prep_features_target(
        val_fold["features"], scaled=config.scaled_targets, feat_keys=config.feat_names
    )

    tr_imgs, tr_targets = shuffle_arrays(train_fold["imgs"], tr_targets, config.seed)
    val_imgs, val_targets = shuffle_arrays(val_fold["imgs"], val_targets, config.seed)

    if config.validation_ratio < 1.0:
        val_limit = int(len(val_imgs) * config.validation_ratio)
        val_imgs = val_imgs[:val_limit]
        val_targets = val_targets[:val_limit]

    train_dataset = CustomDataset(tr_imgs, tr_targets, transform=get_transforms("train"))
    val_dataset = CustomDataset(val_imgs, val_targets, transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, tr_feature_stats


def compute_loss(outputs, targets, criterion, use_awl=False, awl_module=None):
    base_loss = criterion(outputs, targets)
    if not use_awl:
        return base_loss, base_loss

    per_target_losses = [criterion(outputs[:, i], targets[:, i]) for i in range(outputs.shape[1])]
    weighted_loss = awl_module(*per_target_losses)
    mean_loss = torch.stack(per_target_losses).mean()
    return weighted_loss, mean_loss


def train(config: TrainConfig) -> Path:
    set_random_seeds(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if config.split_no not in PANNNUKE_SPLITS:
        raise ValueError(f"Unknown split number {config.split_no}. Valid options: {sorted(PANNNUKE_SPLITS.keys())}")

    train_loader, val_loader, tr_feature_stats = build_dataloaders(config, PANNNUKE_SPLITS[config.split_no])
    frozen = config.freeze_base_epochs > 0
    model = custom_model(
        config.base_model,
        config.num_targets,
        feature_extract=frozen,
        use_pretrained=config.pretrained,
    )
    model = model.to(device)

    criterion = nn.MSELoss()
    awl_module = AutomaticWeightedLoss(config.num_targets).to(device) if config.use_awl else None
    if config.use_awl:
        optimizer = optim.Adam(
            [
                {"params": model.parameters(), "lr": config.learning_rate},
                {"params": awl_module.parameters(), "lr": config.awl_learning_rate, "weight_decay": 0},
            ]
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = (
        lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=0.5)
        if config.use_lr_scheduler
        else None
    )

    best_val_loss = float("inf")
    best_epoch = -1
    history = {"train_loss": [], "val_loss": [], "val_mse": []}
    epochs_without_improvement = 0

    print(f"Saving checkpoints to {config.checkpoint_path}")
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch["x"].to(device)
            labels = batch["y"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, _ = compute_loss(outputs, labels, criterion, config.use_awl, awl_module)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / max(1, len(train_loader))
        history["train_loss"].append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["x"].to(device)
                labels = batch["y"].to(device)
                outputs = model(inputs)
                loss, mse_loss = compute_loss(outputs, labels, criterion, config.use_awl, awl_module)
                val_loss += loss.item()
                val_mse += mse_loss.item()

        val_loss /= max(1, len(val_loader))
        val_mse /= max(1, len(val_loader))
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)

        print(
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"train_loss={epoch_train_loss:.4f} | val_loss={val_loss:.4f} | val_mse={val_mse:.4f}"
        )

        if scheduler:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(model, config, tr_feature_stats, best_val_loss, best_epoch, history)
        else:
            epochs_without_improvement += 1

        if frozen and (epoch + 1) == config.freeze_base_epochs:
            print("Unfreezing backbone parameters.")
            frozen = False
            set_parameter_requires_grad(model.base_model, True)
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * config.lr_decay_on_unfreeze
                print(f"  Updated learning rate to {g['lr']}")

        if epochs_without_improvement > config.early_stop_patience:
            print("Early stopping triggered.")
            break

    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    return config.checkpoint_path


def save_checkpoint(model, config, feature_stats, best_val_loss, best_epoch, history):
    payload = {
        "model_state_dict": model.state_dict(),
        "base_model": config.base_model,
        "num_targets": config.num_targets,
        "feature_names": config.feat_names,
        "feature_stats": feature_stats,
        "config": config.as_dict(),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    torch.save(payload, config.checkpoint_path)

    summary = {
        "config": config.as_dict(),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "feature_stats": feature_stats,
        "history": history,
        "checkpoint": str(config.checkpoint_path),
    }
    with open(config.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Morph-Net on PanNuke")
    parser.add_argument("--data-dir", type=Path, default=Path("data/PanNuke"), help="Root directory of the PanNuke dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory to store checkpoints and logs.")
    parser.add_argument("--device-id", type=str, default="0", help="CUDA device id.")
    parser.add_argument("--split-no", type=int, choices=[1, 2, 3], default=1, help="Cross-validation split number.")
    parser.add_argument("--base-model", choices=["densenet", "resnet"], default="densenet", help="Backbone architecture.")
    parser.add_argument(
        "--feat-types",
        nargs="+",
        default=list(DEFAULT_FEAT_TYPES),
        help="Morphological features to predict (use the defaults from the paper if not provided).",
    )
    parser.add_argument("--model-prefix", type=str, default="MorphNet", help="Prefix used for checkpoint filenames.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the model parameters.")
    parser.add_argument("--awl", action="store_true", help="Enable Automatic Weighted Loss.")
    parser.add_argument("--awl-lr", type=float, default=1e-3, help="Learning rate for AWL parameters.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable loading ImageNet pretrained weights.")
    parser.add_argument("--no-lr-scheduler", action="store_true", help="Disable the step LR scheduler.")
    parser.add_argument("--scheduler-step", type=int, default=25, help="Step size for LR scheduler.")
    parser.add_argument("--validation-ratio", type=float, default=1.0, help="Fraction of validation fold to use.")
    parser.add_argument("--early-stop", type=int, default=100, help="Stop after this many epochs without improvement.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--freeze-base-epochs", type=int, default=0, help="Freeze backbone for the first N epochs.")
    parser.add_argument("--lr-decay-on-unfreeze", type=float, default=0.1, help="LR multiplier applied when unfreezing backbone.")
    parser.add_argument("--no-scale-targets", action="store_true", help="Disable feature scaling for regression targets.")
    return parser.parse_args()


def build_config_from_args(args) -> TrainConfig:
    feat_types = args.feat_types if args.feat_types else list(DEFAULT_FEAT_TYPES)
    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device_id=args.device_id,
        split_no=args.split_no,
        base_model=args.base_model,
        feat_types=feat_types,
        model_name_prefix=args.model_prefix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        awl_learning_rate=args.awl_lr,
        pretrained=not args.no_pretrained,
        use_lr_scheduler=not args.no_lr_scheduler,
        scheduler_step_size=args.scheduler_step,
        validation_ratio=args.validation_ratio,
        early_stop_patience=args.early_stop,
        scaled_targets=not args.no_scale_targets,
        use_awl=args.awl,
        freeze_base_epochs=args.freeze_base_epochs,
        lr_decay_on_unfreeze=args.lr_decay_on_unfreeze,
        seed=args.seed,
    )


if __name__ == "__main__":
    args = parse_args()
    config = build_config_from_args(args)
    train(config)
