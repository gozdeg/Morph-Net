from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import skimage.measure
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def get_transforms(ttype: str):
    """Return albumentations pipelines for the requested split."""
    data_transforms = {
        "train": A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(p=0.25, var_limit=(10.0, 50.0), mean=0),
                A.GaussianBlur(p=0.25),
                A.MedianBlur(p=0.25, blur_limit=5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        "val": A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
    }
    if ttype not in ["train", "val", "ts"]:
        raise ValueError(f"Unsupported transform type: {ttype}")
    return data_transforms["val"] if ttype == "ts" else data_transforms[ttype]


class CustomDataset(Dataset):
    """Simple dictionary-based dataset that keeps images in-memory."""

    def __init__(self, x, y, transform=None, target_transform=None):
        self.x = x
        self.y = torch.from_numpy(y.astype("float32"))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]
        
        if self.transform:
            image = self.transform(image=image)["image"]
        #if self.target_transform:
        #    label = self.target_transform(label)
            
        sample = {"x": image, "y": label}
        return sample
    
##########

def get_arr_by_feature_type(
    arr: np.ndarray, nclasses: int = 4, classid: int | None = None, feat_keys: Sequence[str] = ()
) -> Dict[str, List[float]]:
    """
    Convert flattened network outputs into {feature_name: values} dictionaries.
    """
    npatches = len(arr)
    by_feature_type = {k: [] for k in feat_keys}

    if classid is not None:
        for p in range(npatches):
            for i, key in enumerate(feat_keys):
                by_feature_type[key].append(arr[p, i * nclasses + classid])
    else:
        for p in range(npatches):
            for c in range(nclasses):
                for i, key in enumerate(feat_keys):
                    by_feature_type[key].append(arr[p, i * nclasses + c])

    return by_feature_type


##########
def calculate_features(img, inst_map, feat_names=None):
    """
    Calculate morphological descriptors for a single class mask.
    """
    all_feat_types = [
        "area",
        "major_axis_length",
        "minor_axis_length",
        "perimeter",
        "feret_diameter_max",
        "eccentricity",
    ]
    feats: Dict[str, float] = {}
    props = skimage.measure.regionprops(skimage.measure.label(inst_map), intensity_image=img)

    feats["count"] = len(props)

    for feat_name in all_feat_types:
        total, avg, mn, mx = calculate_single_prop_type(props, feat_name=feat_name)
        feats[f"total_{feat_name}"] = total
        feats[f"avg_{feat_name}"] = avg
        feats[f"min_{feat_name}"] = mn
        feats[f"max_{feat_name}"] = mx

    if feat_names is not None and len(feats.keys()) > len(feat_names):
        feats = {selected_key: feats[selected_key] for selected_key in feat_names}

    return feats

def calculate_single_prop_type(props, feat_name=""):
    all_arr = []
    for region in props:
        all_arr.append(getattr(region, feat_name))

    count = len(all_arr)
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0

    arr = np.asarray(all_arr, dtype=np.float32)
    total = float(np.sum(arr))
    avg = float(total / count)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    return total, avg, mn, mx

##########
'''
def znorm(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    #print(std)
    if(std == 0):     
        std = 1
    newarr = (arr - mean)/std
    
    return newarr
'''

def scale(arr):
    """Scale an array to [0,1]."""
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    newarr = arr - mn
    denom = mx - mn
    if denom != 0:
        newarr = newarr / denom
    return newarr


def prep_features_target(features, scaled=True, feat_keys=None):
    """
    Prepare flattened target vectors for each patch for NN training.

    Args:
        features: list (per patch) of list (per class) of feature-dictionaries.
        scaled: whether to scale each feature type to [0,1].
        feat_keys: optional feature subset. If None, use keys from the first entry.

    Returns:
        target_vectors, scaled_by_feature_type, feature_stats
    """
    npatches = len(features)
    nclasses = len(features[0])
    if not feat_keys:
        feat_keys = list(features[0][0].keys())

    aggregated: Dict[str, List[float]] = {k: [] for k in feat_keys}
    for patch in features:
        for c in range(nclasses):
            for k in feat_keys:
                aggregated[k].append(patch[c][k])

    feature_stats: Dict[str, Dict[str, float]] = {}
    scaled_by_feature_type: Dict[str, np.ndarray] = {}

    for k in feat_keys:
        raw_arr = np.asarray(aggregated[k], dtype=np.float32)
        feature_stats[k] = {"min": float(np.min(raw_arr)), "max": float(np.max(raw_arr))}
        scaled_values = scale(raw_arr) if scaled else raw_arr
        scaled_by_feature_type[k] = scaled_values

    target_vectors = np.zeros([npatches, nclasses * len(feat_keys)], dtype=np.float32)
    for p in range(npatches):
        vec = []
        for k in feat_keys:
            vec.append(scaled_by_feature_type[k][p * nclasses : (p + 1) * nclasses])
        target_vectors[p, :] = np.asarray(vec).flatten()

    return target_vectors, scaled_by_feature_type, feature_stats
