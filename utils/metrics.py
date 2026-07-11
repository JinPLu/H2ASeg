"""Segmentation metrics used by the H2ASeg training and inference scripts."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


SMOOTH = 1e-3


def _foreground_channel(array):
    """Select the foreground channel from a two-channel mask."""
    if array.ndim < 2:
        raise ValueError(f"Expected a batched channel-first mask, got shape {array.shape}")
    return array[:, 1:] if array.shape[1] == 2 else array


def _validate_shapes(gt, pred) -> None:
    if gt.shape != pred.shape:
        raise ValueError(f"Ground truth and prediction shapes differ: {gt.shape} != {pred.shape}")


def show_deep_metrics(
    outputs: Sequence[torch.Tensor], labels: torch.Tensor, deep: bool = True
):
    """Print metrics for deep-supervision outputs and return the first output's metrics."""
    selected_outputs = outputs if deep else outputs[:1]
    first_metrics = None
    for index, output in enumerate(selected_outputs):
        prediction = output.argmax(dim=1, keepdim=True)
        result = metrics_tensor(labels, prediction)
        suffix = "\n" if index == len(selected_outputs) - 1 else ""
        print(
            "[FP:{:.4f}, FN:{:.4f}, IoU:{:.4f}, Dice:{:.4f} "
            "pix:{:6}/{:6}]{}".format(
                *result,
                int(prediction.sum().item()),
                int(labels.sum().item()),
                suffix,
            )
        )
        if first_metrics is None:
            first_metrics = list(result)
    if first_metrics is None:
        raise ValueError("outputs must contain at least one prediction tensor")
    return first_metrics


def get_hausdorff(gt, pred) -> float:
    """Return HD95, or NaN when either binary mask is empty.

    MedPy is imported lazily so Dice/IoU evaluation does not require it.
    """
    gt_array = gt.detach().cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)
    pred_array = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
    if not np.any(gt_array) or not np.any(pred_array):
        return float("nan")

    from medpy.metric.binary import hd95

    return float(hd95(pred_array, gt_array))


def metrics(gt, pred):
    """Compute false-positive rate, false-negative rate, IoU and Dice with NumPy."""
    gt_array = _foreground_channel(np.asarray(gt)).astype(bool, copy=False)
    pred_array = _foreground_channel(np.asarray(pred)).astype(bool, copy=False)
    _validate_shapes(gt_array, pred_array)

    intersection = np.logical_and(gt_array, pred_array)
    union = np.logical_or(gt_array, pred_array)
    false_positive = np.logical_and(pred_array, np.logical_not(gt_array)).sum()
    false_negative = np.logical_and(gt_array, np.logical_not(pred_array)).sum()
    true_positive = intersection.sum()
    true_negative = np.logical_not(union).sum()
    gt_sum = gt_array.sum()
    pred_sum = pred_array.sum()

    false_positive_rate = false_positive / (false_positive + true_negative + SMOOTH)
    false_negative_rate = false_negative / (false_negative + true_positive + SMOOTH)
    jaccard = true_positive / (union.sum() + SMOOTH)
    dice = 2 * true_positive / (gt_sum + pred_sum + SMOOTH)

    return tuple(
        float(value)
        for value in (false_positive_rate, false_negative_rate, jaccard, dice)
    )


def metrics_tensor(gt: torch.Tensor, pred: torch.Tensor):
    """Compute metrics without moving tensors away from their current device."""
    return metrics_tensor_one_patient(gt, pred)


def metrics_tensor_one_patient(gt: torch.Tensor, pred: torch.Tensor):
    """Tensor implementation of :func:`metrics` for one batch or patient."""
    if gt.device != pred.device:
        raise ValueError(f"Ground truth and prediction devices differ: {gt.device} != {pred.device}")

    gt_mask = _foreground_channel(gt).bool()
    pred_mask = _foreground_channel(pred).bool()
    _validate_shapes(gt_mask, pred_mask)

    intersection = torch.logical_and(gt_mask, pred_mask)
    union = torch.logical_or(gt_mask, pred_mask)
    false_positive = torch.logical_and(pred_mask, torch.logical_not(gt_mask)).sum()
    false_negative = torch.logical_and(gt_mask, torch.logical_not(pred_mask)).sum()
    true_positive = intersection.sum()
    true_negative = torch.logical_not(union).sum()
    gt_sum = gt_mask.sum()
    pred_sum = pred_mask.sum()

    false_positive_rate = false_positive / (false_positive + true_negative + SMOOTH)
    false_negative_rate = false_negative / (false_negative + true_positive + SMOOTH)
    jaccard = true_positive / (union.sum() + SMOOTH)
    dice = 2 * true_positive / (gt_sum + pred_sum + SMOOTH)

    return tuple(
        float(value.item())
        for value in (false_positive_rate, false_negative_rate, jaccard, dice)
    )


def metrics_tensor_one_patience(gt: torch.Tensor, pred: torch.Tensor):
    """Backward-compatible alias for the historical misspelled function name."""
    return metrics_tensor_one_patient(gt, pred)
