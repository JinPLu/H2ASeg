"""Segmentation metrics used by the H2ASeg training and inference scripts."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:  # Keep the NumPy metrics usable without PyTorch.
    if exc.name != "torch":
        raise
    torch = None


SMOOTH = 1e-3


def _validate_mask_dimensions(array, name: str) -> None:
    """Validate the shared ``[batch, channel, *spatial]`` mask contract."""
    if array.ndim < 3:
        raise ValueError(
            f"{name} must be a batched channel-first mask with at least one "
            f"spatial dimension, got shape {array.shape}"
        )
    if array.shape[1] not in (1, 2):
        raise ValueError(
            f"{name} must have one label channel or two one-hot channels, "
            f"got shape {array.shape}"
        )
    if any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must not contain zero-sized dimensions: {array.shape}")


def _foreground_channel(array, name: str):
    """Select the foreground channel from a validated binary mask."""
    _validate_mask_dimensions(array, name)
    return array[:, 1:] if array.shape[1] == 2 else array


def _validate_shapes(gt, pred) -> None:
    if gt.shape != pred.shape:
        raise ValueError(f"Ground truth and prediction shapes differ: {gt.shape} != {pred.shape}")


def _numpy_mask(array, name: str) -> np.ndarray:
    array = np.asarray(array)
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError(f"{name} must contain numeric or boolean values, got {array.dtype}")
    if np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return _foreground_channel(array, name).astype(bool, copy=False)


def _numpy_ratio(numerator: int, denominator: int, *, empty_value: float = 0.0) -> float:
    if denominator == 0:
        return empty_value
    return float(numerator / (denominator + SMOOTH))


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for tensor metrics")


def _tensor_mask(array, name: str):
    _require_torch()
    if not torch.is_tensor(array):
        raise TypeError(f"{name} must be a torch.Tensor")
    _validate_mask_dimensions(array, name)
    if (torch.is_floating_point(array) or torch.is_complex(array)) and not bool(
        torch.isfinite(array).all().item()
    ):
        raise ValueError(f"{name} contains NaN or infinite values")
    return _foreground_channel(array, name).bool()


def _tensor_ratio(numerator, denominator, *, empty_value: float = 0.0):
    # float64 keeps large 3-D voxel counts precise on the supported CPU/CUDA paths.
    numerator = numerator.to(dtype=torch.float64)
    denominator = denominator.to(dtype=torch.float64)
    value = numerator / (denominator + SMOOTH)
    return torch.where(
        denominator == 0,
        torch.full_like(value, empty_value),
        value,
    )


def show_deep_metrics(
    outputs: Sequence[torch.Tensor], labels: torch.Tensor, deep: bool = True
):
    """Print metrics for deep-supervision outputs and return the first output's metrics."""
    _require_torch()
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
    gt_array = (
        gt.detach().cpu().numpy()
        if torch is not None and torch.is_tensor(gt)
        else np.asarray(gt)
    )
    pred_array = (
        pred.detach().cpu().numpy()
        if torch is not None and torch.is_tensor(pred)
        else np.asarray(pred)
    )
    _validate_shapes(gt_array, pred_array)
    if (
        np.issubdtype(gt_array.dtype, np.inexact)
        and not np.isfinite(gt_array).all()
    ) or (
        np.issubdtype(pred_array.dtype, np.inexact)
        and not np.isfinite(pred_array).all()
    ):
        raise ValueError("Hausdorff masks must contain only finite values")
    if not np.any(gt_array) or not np.any(pred_array):
        return float("nan")

    try:
        from medpy.metric.binary import hd95
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "get_hausdorff requires the optional 'medpy' package"
        ) from exc

    return float(hd95(pred_array, gt_array))


def metrics(gt, pred):
    """Compute false-positive rate, false-negative rate, IoU and Dice with NumPy.

    Empty ground truth and prediction masks are a correct match (IoU and Dice
    equal one).  If exactly one mask is empty, both overlap metrics equal zero.
    """
    gt_array = _numpy_mask(gt, "Ground truth")
    pred_array = _numpy_mask(pred, "Prediction")
    _validate_shapes(gt_array, pred_array)

    intersection = np.logical_and(gt_array, pred_array)
    union = np.logical_or(gt_array, pred_array)
    false_positive = int(np.logical_and(pred_array, np.logical_not(gt_array)).sum())
    false_negative = int(np.logical_and(gt_array, np.logical_not(pred_array)).sum())
    true_positive = int(intersection.sum())
    true_negative = int(np.logical_not(union).sum())
    gt_sum = int(gt_array.sum())
    pred_sum = int(pred_array.sum())
    union_sum = int(union.sum())

    false_positive_rate = _numpy_ratio(
        false_positive, false_positive + true_negative
    )
    false_negative_rate = _numpy_ratio(
        false_negative, false_negative + true_positive
    )
    jaccard = _numpy_ratio(true_positive, union_sum, empty_value=1.0)
    dice = _numpy_ratio(2 * true_positive, gt_sum + pred_sum, empty_value=1.0)

    return false_positive_rate, false_negative_rate, jaccard, dice


def metrics_tensor(gt: torch.Tensor, pred: torch.Tensor):
    """Compute metrics without moving tensors away from their current device."""
    return metrics_tensor_one_patient(gt, pred)


def metrics_tensor_one_patient(gt: torch.Tensor, pred: torch.Tensor):
    """Tensor implementation of :func:`metrics` for one batch or patient."""
    _require_torch()
    if not torch.is_tensor(gt) or not torch.is_tensor(pred):
        raise TypeError("Ground truth and prediction must both be torch.Tensor instances")
    if gt.device != pred.device:
        raise ValueError(f"Ground truth and prediction devices differ: {gt.device} != {pred.device}")

    gt_mask = _tensor_mask(gt, "Ground truth")
    pred_mask = _tensor_mask(pred, "Prediction")
    _validate_shapes(gt_mask, pred_mask)

    intersection = torch.logical_and(gt_mask, pred_mask)
    union = torch.logical_or(gt_mask, pred_mask)
    false_positive = torch.logical_and(pred_mask, torch.logical_not(gt_mask)).sum()
    false_negative = torch.logical_and(gt_mask, torch.logical_not(pred_mask)).sum()
    true_positive = intersection.sum()
    true_negative = torch.logical_not(union).sum()
    gt_sum = gt_mask.sum()
    pred_sum = pred_mask.sum()

    false_positive_rate = _tensor_ratio(
        false_positive, false_positive + true_negative
    )
    false_negative_rate = _tensor_ratio(
        false_negative, false_negative + true_positive
    )
    jaccard = _tensor_ratio(true_positive, union.sum(), empty_value=1.0)
    dice = _tensor_ratio(2 * true_positive, gt_sum + pred_sum, empty_value=1.0)

    return tuple(
        float(value.item())
        for value in (false_positive_rate, false_negative_rate, jaccard, dice)
    )


def metrics_tensor_one_patience(gt: torch.Tensor, pred: torch.Tensor):
    """Backward-compatible alias for the historical misspelled function name."""
    return metrics_tensor_one_patient(gt, pred)
