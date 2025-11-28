from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from dataset_utils import calculate_features

# Splits in the Pannuke dataset
PANNNUKE_SPLITS = {
    1: {"tr": 1, "val": 2, "ts": 3},
    2: {"tr": 2, "val": 1, "ts": 3},
    3: {"tr": 3, "val": 2, "ts": 1},
}


def _resolve(base: Path | str, *parts: str) -> Path:
    base_path = Path(base).expanduser()
    return base_path.joinpath(*parts)


def _require_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {path}")
    return path


def get_pannuke_fold_data(foldno: int, data_dir: Path | str) -> Dict[str, np.ndarray]:
    """
    Load raw arrays for a single PanNuke fold.

    Args:
        foldno: Fold number in {1,2,3}.
        data_dir: Root directory containing Fold*/images and Fold*/masks subfolders.

    Returns:
        Dictionary with keys imgs, types, masks.
    """
    print(f"Loading PanNuke fold {foldno} from {data_dir}")
    fold_root = _resolve(data_dir, f"Fold{foldno}")
    images_path = _require_exists(fold_root / "images" / f"fold{foldno}" / "images.npy")
    types_path = _require_exists(fold_root / "images" / f"fold{foldno}" / "types.npy")
    masks_path = _require_exists(fold_root / "masks" / f"fold{foldno}" / "masks.npy")

    data: Dict[str, np.ndarray] = {}
    data["imgs"] = np.load(images_path)
    data["imgs"] = data["imgs"].astype(np.uint8, copy=False)
    data["types"] = np.load(types_path)
    raw_masks = np.load(masks_path)
    data["masks"] = convert_masks_to_four_classes(raw_masks)

    print(f"  imgs shape:   {data['imgs'].shape}")
    print(f"  types shape:  {data['types'].shape}")
    print(f"  masks shape:  {data['masks'].shape}")
    print("  converted masks.npy -> 4 classes (Neoplastic/Inflammatory/Connective/Epithelial)")
    return data


def get_postprep_data(
    foldno: int,
    feat_names: Sequence[str] | None = None,
    data_dir: Path | str = "data/PanNuke",
) -> Dict[str, List[List[Dict[str, float]]]]:
    """
    Load a fold and pre-compute morphological features for each image/class channel.

    Args:
        foldno: Fold number in {1,2,3}.
        feat_names: Optional subset of feature keys to keep. When None all features are returned.
        data_dir: Root data directory (see README for expected layout).

    Returns:
        Dictionary with the raw arrays plus ``features`` list.
    """
    nclasses = 4
    folddata = get_pannuke_fold_data(foldno, data_dir)
    folddata["features"] = []
    nimgs = len(folddata["imgs"])

    for imno in range(nimgs):
        img_feats: List[Dict[str, float]] = []
        for class_idx in range(nclasses):
            feats = calculate_features(
                folddata["imgs"][imno],
                folddata["masks"][imno, :, :, class_idx],
                feat_names=feat_names,
            )
            img_feats.append(feats)
        folddata["features"].append(img_feats)

    return folddata


def convert_masks_to_four_classes(mask_arr: np.ndarray) -> np.ndarray:
    """
    Convert original PanNuke masks (5 channels + background) into 4-class format.

    Original ordering: 0=Neoplastic, 1=Inflammatory, 2=Connective,
    3=Dead, 4=Epithelial, 5=Background. We discard dead cells/background
    and copy the Epithelial channel into index 3.
    """
    converted = np.zeros(mask_arr.shape[:-1] + (4,), dtype=mask_arr.dtype)
    converted[..., 0:3] = mask_arr[..., 0:3]
    converted[..., 3] = mask_arr[..., 4]
    return converted

 
