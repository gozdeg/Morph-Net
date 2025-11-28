from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn import metrics

from dataset_utils import get_arr_by_feature_type
from models import custom_model

NCLASSES = 4


def load_checkpoint(checkpoint_path: Path | str, device: torch.device):
    """
    Load a trained Morph-Net checkpoint saved by ``train_pannuke.py``.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = custom_model(
        checkpoint["base_model"],
        checkpoint["num_targets"],
        feature_extract=False,
        use_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def run_inference(model: torch.nn.Module, dataloader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["x"].to(device)
            labels = batch["y"].to(device)
            outputs = model(inputs)
            preds.append(outputs.cpu())
            targets.append(labels.cpu())

    all_preds = torch.cat(preds, dim=0).numpy()
    all_targets = torch.cat(targets, dim=0).numpy()
    return all_preds, all_targets


def _rescale(values: Dict[str, Sequence[float]], feature_stats: Dict[str, Dict[str, float]], feat_names):
    if feature_stats is None:
        return {k: np.asarray(v, dtype=np.float32) for k, v in values.items()}

    scaled = {}
    for k in feat_names:
        arr = np.asarray(values[k], dtype=np.float32)
        tr_max = feature_stats[k]["max"]
        if tr_max != 0:
            arr = arr * tr_max
        scaled[k] = arr
    return scaled


def results(all_preds, all_targets, feat_names, tr_feature_stats=None, rescale=True, target_scaled=False):
    preds_by_ft = get_arr_by_feature_type(all_preds, feat_keys=feat_names)
    targets_by_ft = get_arr_by_feature_type(all_targets, feat_keys=feat_names)

    if rescale and tr_feature_stats:
        preds_by_ft = _rescale(preds_by_ft, tr_feature_stats, feat_names)
        if target_scaled:
            targets_by_ft = _rescale(targets_by_ft, tr_feature_stats, feat_names)
    else:
        preds_by_ft = {k: np.asarray(v, dtype=np.float32) for k, v in preds_by_ft.items()}
        targets_by_ft = {k: np.asarray(v, dtype=np.float32) for k, v in targets_by_ft.items()}

    summary = {}
    for k in feat_names:
        y_true = targets_by_ft[k]
        y_pred = preds_by_ft[k]
        rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
        r2 = metrics.r2_score(y_true, y_pred)
        srho, spval = stats.spearmanr(y_true, y_pred)
        prho, ppval = stats.pearsonr(y_true, y_pred)
        summary[k] = {
            "rmse": float(rmse),
            "r2": float(r2),
            "spearman": float(srho),
            "spearman_p": float(spval),
            "pearson": float(prho),
            "pearson_p": float(ppval),
        }
    return summary


def results_classwise(all_preds, all_targets, feat_names, tr_feature_stats=None, rescale=True, target_scaled=False):
    class_results: Dict[str, Dict[int, Dict[str, float]]] = {}
    for classid in range(NCLASSES):
        class_targets = get_arr_by_feature_type(all_targets, classid=classid, feat_keys=feat_names)
        class_preds = get_arr_by_feature_type(all_preds, classid=classid, feat_keys=feat_names)

        if rescale and tr_feature_stats:
            class_preds = _rescale(class_preds, tr_feature_stats, feat_names)
            if target_scaled:
                class_targets = _rescale(class_targets, tr_feature_stats, feat_names)
        else:
            class_preds = {k: np.asarray(v, dtype=np.float32) for k, v in class_preds.items()}
            class_targets = {k: np.asarray(v, dtype=np.float32) for k, v in class_targets.items()}

        for feat_name in feat_names:
            metrics_dict = class_results.setdefault(feat_name, {})
            y_true = class_targets[feat_name]
            y_pred = class_preds[feat_name]
            rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
            r2 = metrics.r2_score(y_true, y_pred)
            srho, spval = stats.spearmanr(y_true, y_pred)
            prho, ppval = stats.pearsonr(y_true, y_pred)
            metrics_dict[classid] = {
                "rmse": float(rmse),
                "r2": float(r2),
                "spearman": float(srho),
                "spearman_p": float(spval),
                "pearson": float(prho),
                "pearson_p": float(ppval),
            }

    return class_results
