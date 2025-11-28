# Morph-Net

Code accompanying the workshop paper **“Morph-Net: End-to-End Prediction of Nuclear Morphological Features from Histology Images.”**

Morph-Net directly regresses morphological descriptors (area, major/minor axis length, perimeter, eccentricity, and instance counts) for four nucleus categories without relying on intermediate segmentation masks. This repository contains a lightweight training and evaluation pipeline for the PanNuke dataset together with cleaned utility scripts and notebooks.

## Repository layout

- `train_pannuke.py` – main training script with a CLI.
- `evaluate_pannuke.py` – evaluation/metric reporting for saved checkpoints.
- `dataset_pannuke.py`, `dataset_utils.py`, `models.py` – core data/model helpers.
- `AutomaticWeightedLoss.py`, `gradcam.py`, `gradcam_utils.py`, `gpu_utils.py` – optional utilities.
- `demo_train.py` / `demo_evaluate.py` – simple scripts covering the typical training/evaluation workflow.
- `gradcam_demo.ipynb` – minimal Grad-CAM visualization notebook built on top of the cleaned code.
- `data/` – placeholder for locally downloaded PanNuke `.npy` files (see `data/README.md`).
- `output/` – default output directory (git-ignored, `.gitkeep` keeps it visible).

Legacy `.npy` result dumps, checkpoints, and temporary folders have been removed for clarity.

## Dataset

Download the PanNuke folds and unpack them under `data/PanNuke` so that each fold contains:

```
Fold{N}/images/fold{N}/images.npy
Fold{N}/images/fold{N}/types.npy
Fold{N}/masks/fold{N}/masks.npy
```

This layout is reflected in `data/README.md`. 

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements capture the core runtime (PyTorch, albumentations, scikit-image, etc.). Notebooks may require additional packages such as `pandas` or `matplotlib`.

## Training

```
python train_pannuke.py \
  --data-dir data/PanNuke \
  --output-dir output \
  --split-no 1 \
  --device-id 0
```

Key options:

- `--feat-types` controls the morphological properties to regress (defaults to the paper configuration). The `count` channel is always included.
- `--awl` enables Automatic Weighted Loss jointly with `--awl-lr`.
- `--freeze-base-epochs` freezes the backbone for the first N epochs before unfreezing and decaying the LR.
- `--no-scale-targets` disables feature scaling if raw magnitudes are desired.

Each run writes an `*.pt` checkpoint and a JSON summary (loss curves, config, feature stats) into `output/`. These stats are used automatically during evaluation to convert predictions back to real-world units.

## Evaluation

```
python evaluate_pannuke.py \
  --checkpoint output/MorphNet-split1-densenet-area-major_axis_length-minor_axis_length-perimeter-eccentricity-scaled.pt \
  --data-dir data/PanNuke \
  --subset test \
  --per-class
```

The script reports RMSE/R²/Pearson/Spearman for each feature and optionally per-class. Use `--target-scaled` if you want to keep the evaluation targets in the normalized space (otherwise the predictions are rescaled via the stored training statistics). Metrics can be exported to JSON via `--save-json`.

## Demo scripts

- `python demo_train.py`: mirrors the hyper-parameters from the original training notebook (DenseNet-121 backbone, Automatic Weighted Loss, batch size 128, 1000 epochs). Pass `--epochs` to shorten experiments.
- `python demo_evaluate.py`: evaluates the demo checkpoint on a chosen subset (`--subset tr|val|test`) using the helper from `evaluate_pannuke.py`.

These scripts are intentionally lightweight entry points that simply call into the main training/evaluation utilities with sensible defaults.

## Notebook

`gradcam_demo.ipynb` demonstrates how to import a saved checkpoint, pick an image/feature/class combination, and visualize Grad-CAM heatmaps with the current codebase. Update the configuration cell with your dataset/checkpoint paths before running.

## Notes

- `.gitignore` shields large artifacts (`output/`, `tmp/`, datasets) to keep the repo reproducible.
- The code base avoids packaging; everything can be executed directly with CLI scripts (`train_pannuke.py`, `evaluate_pannuke.py`, `demo_*.py`).
