"""Train H2ASeg on aligned PET/CT NIfTI volumes.

The MICCAI 2024 model, losses, and default hyperparameters are unchanged.  This
entry point adds explicit data contracts, reproducible splits, and resumable
training state around the original implementation.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import monai
import numpy as np
import torch
from monai.data import DataLoader, list_data_collate
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotated,
    SpatialPadd,
)
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from model.H2ASeg import H2ASeg
from utils.data import (
    ValidateGeometryd,
    discover_cases,
    foreground_above_minimum,
    split_cases,
)
from utils.metrics import metrics_tensor, show_deep_metrics
from utils.model_config import DEFAULT_MODEL_CONFIG, H2ASegModelConfig


DEFAULT_LOSS_WEIGHTS = (5, 4, 3, 2, 1)
CHECKPOINT_FORMAT_VERSION = 2
CHECKPOINT_ARTIFACT_TYPE = "h2aseg_training_checkpoint"
VALIDATION_METRIC_POLICY = "positive_gt_overlap_negative_gt_fpr_v1"
FULL_RESUME_KEYS = frozenset(
    {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "epoch",
        "iteration",
        "model_config",
        "config",
        "best_dice",
        "best_train_dice",
        "resolved_cases",
        "resolved_split",
        "rng_state",
    }
)
VERSIONED_FULL_RESUME_KEYS = FULL_RESUME_KEYS | frozenset(
    {
        "checkpoint_format_version",
        "artifact_type",
        "amp_enabled",
        "validation_metric_policy",
    }
)
TRAINING_STATE_HINT_KEYS = frozenset(
    {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "epoch",
        "iteration",
        "config",
        "best_dice",
        "best_train_dice",
        "rng_state",
        "resolved_cases",
        "resolved_split",
    }
)
RESUME_CONFIG_KEYS = (
    "batch_size",
    "num_workers",
    "seed",
    "lr",
    "split_mode",
    "split_seed",
    "val_interval",
    "val_sw_batch_size",
    "val_overlap",
)
RESUME_METADATA_KEYS = (
    "checkpoint_format_version",
    "artifact_type",
    "epoch",
    "iteration",
    "best_dice",
    "best_train_dice",
    "amp_enabled",
    "validation_metric_policy",
)


@dataclass(frozen=True)
class ValidationSummary:
    """Paper-aligned validation cohorts kept separate for model selection."""

    positive_mean: np.ndarray
    positive_cases: int
    negative_cases: int
    negative_mean_fpr: float | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train H2ASeg on paired PET/CT NIfTI data.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV with case,ct,pet,label[,split,group] columns.",
    )
    parser.add_argument("--ct-dir", type=Path, help="Directory containing CT NIfTI volumes.")
    parser.add_argument("--pet-dir", type=Path, help="Directory containing PET NIfTI volumes.")
    parser.add_argument("--label-dir", type=Path, help="Directory containing label NIfTI volumes.")
    parser.add_argument(
        "--split-mode",
        choices=("legacy", "seeded"),
        default="legacy",
        help="60/20/20 split policy when the manifest has no split column.",
    )
    parser.add_argument("--split-seed", type=int, default=12345)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: save/hecktor2022_H2ASeg/YYYYMMDD_HHMMSS_microseconds).",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=10)
    parser.add_argument("--val-sw-batch-size", type=int, default=4)
    parser.add_argument("--val-overlap", type=float, default=0.5)
    parser.add_argument("--resume", type=Path, help="Full training checkpoint or legacy state_dict.")
    parser.add_argument(
        "--resume-mode",
        choices=("auto", "full", "weights-only"),
        default="auto",
        help=(
            "Resume policy: auto accepts complete full checkpoints and legacy raw "
            "weights; full requires all training state; weights-only explicitly "
            "discards optimizer/scheduler/epoch state."
        ),
    )
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision.")
    args = parser.parse_args(argv)

    directories = (args.ct_dir, args.pet_dir, args.label_dir)
    if args.manifest is not None and any(path is not None for path in directories):
        parser.error("use either --manifest or --ct-dir/--pet-dir/--label-dir, not both")
    if args.manifest is None and any(path is None for path in directories):
        parser.error("provide --manifest or all of --ct-dir, --pet-dir, and --label-dir")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    monai.utils.set_determinism(seed=seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({requested}) but CUDA is unavailable.")
    return device


def create_grad_scaler(enabled: bool):
    """Use the current AMP API while retaining PyTorch 2.1 compatibility."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_transforms(patch_size: tuple[int, int, int]) -> tuple[Compose, Compose]:
    """Build historical training sampling and deterministic full-volume validation."""

    def load_and_validate(*, validate_pet_crop: bool) -> list[Any]:
        # Build independent transform instances: MONAI transforms can carry
        # mutable/random state and must not be shared across train/validation.
        return [
            LoadImaged(keys=["pet", "ct", "label"]),
            EnsureChannelFirstd(keys=["pet", "ct", "label"]),
            ValidateGeometryd(
                keys=["ct", "pet", "label"],
                reference_key="ct",
                case_key="case",
                validate_pet_crop=validate_pet_crop,
            ),
        ]

    train = Compose(
        load_and_validate(validate_pet_crop=True)
        + [
            CropForegroundd(
                keys=["pet", "ct", "label"],
                source_key="pet",
                select_fn=foreground_above_minimum,
                allow_smaller=True,
            ),
            SpatialPadd(keys=["pet", "ct", "label"], spatial_size=patch_size),
            RandCropByPosNegLabeld(
                keys=["pet", "ct", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=2,
            ),
            RandRotated(
                keys=["pet", "ct", "label"],
                prob=0.8,
                range_z=np.pi / 4,
                mode=("bilinear", "bilinear", "nearest"),
            ),
        ]
    )
    # Validation and inference both evaluate the complete aligned source volume.
    val = Compose(load_and_validate(validate_pet_crop=False))
    return train, val


def _json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def resume_training_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return trajectory-affecting CLI values that must match for full resume."""
    values = vars(args)
    return {key: values[key] for key in RESUME_CONFIG_KEYS if key in values}


def validate_resume_config(
    saved: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Reject full-resume CLI changes that would silently alter the run."""
    conflicts = [
        f"{key}: checkpoint={saved[key]!r}, CLI={current[key]!r}"
        for key in RESUME_CONFIG_KEYS
        if key in saved and key in current and saved[key] != current[key]
    ]
    if conflicts:
        raise ValueError(
            "Full resume configuration conflicts with the checkpoint; use the "
            "checkpoint values or --resume-mode weights-only: " + "; ".join(conflicts)
        )


def should_validate_epoch(epoch: int, total_epochs: int, interval: int) -> bool:
    """Validate on the configured cadence and always on the final epoch."""
    return (epoch + 1) % interval == 0 or epoch + 1 == total_epochs


def summarize_validation_cases(
    cases: Sequence[tuple[bool, Sequence[float]]],
) -> ValidationSummary:
    """Aggregate overlap only for positive GT and negative-case FPR separately."""
    positive_metrics: list[np.ndarray] = []
    negative_fprs: list[float] = []
    for has_foreground, values in cases:
        metrics = np.asarray(values, dtype=np.float64)
        if metrics.shape != (4,) or not np.isfinite(metrics).all():
            raise ValueError(
                "Each validation result must contain four finite values: "
                "FPR, FNR, IoU, and Dice."
            )
        if has_foreground:
            positive_metrics.append(metrics)
        else:
            negative_fprs.append(float(metrics[0]))

    if not positive_metrics:
        raise ValueError(
            "Validation split contains no positive-GT cases; Dice/IoU model "
            "selection is undefined. Provide at least one lesion-positive "
            "validation case."
        )

    return ValidationSummary(
        positive_mean=np.mean(positive_metrics, axis=0),
        positive_cases=len(positive_metrics),
        negative_cases=len(negative_fprs),
        negative_mean_fpr=(
            float(np.mean(negative_fprs)) if negative_fprs else None
        ),
    )


def resolve_resumed_best_dice(metadata: Mapping[str, Any]) -> float:
    """Reuse best Dice only when its validation cohort policy is comparable."""
    best_dice = float(metadata.get("best_dice", 0.0))
    if not np.isfinite(best_dice):
        raise ValueError("Checkpoint best_dice must be finite.")
    saved_policy = metadata.get("validation_metric_policy")
    if metadata and saved_policy != VALIDATION_METRIC_POLICY:
        warnings.warn(
            "Checkpoint best_dice used a missing or incompatible validation "
            "metric policy; resetting best_dice to 0 so positive-GT model "
            "selection can create a valid val_best artifact.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0
    return best_dice


def binarize_labels(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Map every non-zero source label to foreground before integer conversion."""
    return (labels.to(device=device, non_blocking=True) != 0).long()


def ensure_finite_losses(loss: torch.Tensor, components: Sequence[torch.Tensor]) -> None:
    """Fail before backward/optimizer mutation when an objective is non-finite."""
    finite = bool(torch.isfinite(loss.detach()).all()) and all(
        bool(torch.isfinite(value.detach()).all()) for value in components
    )
    if finite:
        return
    values = [float(value.detach().cpu()) for value in components]
    raise FloatingPointError(
        f"Non-finite training loss detected: total={float(loss.detach().cpu())}, "
        f"components={values}"
    )


def predict_validation_logits(
    model: nn.Module,
    ct: torch.Tensor,
    pet: torch.Tensor,
    *,
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: float,
    model_device: torch.device,
) -> torch.Tensor:
    """Run patches on the model device while keeping full volumes/logits on CPU."""
    inputs = torch.cat(
        (
            ct.to(device="cpu", dtype=torch.float32),
            pet.to(device="cpu", dtype=torch.float32),
        ),
        dim=1,
    )

    def predict_patch(patch: torch.Tensor) -> torch.Tensor:
        return model(patch[:, :1], patch[:, 1:2])[0]

    return sliding_window_inference(
        inputs,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=predict_patch,
        overlap=overlap,
        sw_device=model_device,
        device=torch.device("cpu"),
    )


def _copy_cases(cases: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [dict(case) for case in cases]


def resolved_split_snapshot(
    train_cases: Sequence[Mapping[str, str]],
    val_cases: Sequence[Mapping[str, str]],
    test_cases: Sequence[Mapping[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Create checkpoint-safe copies of the exact resolved partitions."""
    return {
        "train": _copy_cases(train_cases),
        "val": _copy_cases(val_cases),
        "test": _copy_cases(test_cases),
    }


def write_resolved_split(
    path: Path, resolved_split: Mapping[str, Sequence[Mapping[str, str]]]
) -> None:
    """Persist the exact train/validation/test assignment used by a run."""
    records = [
        case
        for split_name in ("train", "val", "test")
        for case in resolved_split.get(split_name, ())
    ]
    group_flags = ["group" in case for case in records]
    if any(group_flags) and not all(group_flags):
        raise ValueError(
            "Resolved split group must be present for every case or for none."
        )
    include_group = bool(group_flags and all(group_flags))
    if include_group and any(
        not str(case.get("group", "")).strip() for case in records
    ):
        raise ValueError("Resolved split group values must be non-empty.")
    fieldnames = ["case", "ct", "pet", "label"]
    if include_group:
        fieldnames.append("group")
    fieldnames.append("split")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            for case in resolved_split.get(split_name, ()):
                row = {
                    "case": case["case"],
                    "ct": case["ct"],
                    "pet": case["pet"],
                    "label": case["label"],
                    "split": split_name,
                }
                if include_group:
                    row["group"] = case["group"]
                writer.writerow(row)


def validate_resume_split(
    saved: Mapping[str, Sequence[Mapping[str, str]]],
    current: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    """Prevent a full-state resume from silently changing its data assignment."""
    def normalize(
        partitions: Mapping[str, Sequence[Mapping[str, str]]]
    ) -> dict[str, list[dict[str, str]]]:
        result: dict[str, list[dict[str, str]]] = {}
        for name in ("train", "val", "test"):
            result[name] = []
            for original in partitions.get(name, ()):
                case = dict(original)
                redundant_split = case.pop("split", None)
                if (
                    redundant_split is not None
                    and str(redundant_split).strip().casefold() != name
                ):
                    raise ValueError(
                        f"Resolved split record {case.get('case')!r} is stored in "
                        f"partition {name!r} but declares split {redundant_split!r}."
                    )
                result[name].append(case)
        return result

    normalized_saved = normalize(saved)
    normalized_current = normalize(current)
    if normalized_saved != normalized_current:
        raise ValueError(
            "The current resolved data split differs from the full checkpoint. "
            "Use the checkpoint's data/manifest for exact resume, or load a "
            "legacy model-only .pth for transfer learning."
        )


def _numpy_rng_state_to_dict(state: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "bit_generator": state[0],
        "keys": state[1].tolist(),
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _numpy_rng_state_from_dict(state: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(state["bit_generator"]),
        np.asarray(state["keys"], dtype=np.uint32),
        int(state["position"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    )


def _iter_transform_random_states(transform: Any):
    """Yield stable paths to MONAI Randomizable RNGs, including crop helpers."""
    seen: set[int] = set()

    def visit(value: Any, path: str):
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        rng = getattr(value, "R", None)
        if rng is not None and hasattr(rng, "get_state"):
            yield path, value, rng
        transforms = getattr(value, "transforms", None)
        if transforms is not None:
            for index, child in enumerate(transforms):
                yield from visit(child, f"{path}.transforms[{index}]")
        for attribute in ("cropper", "rand_rotate", "rand_affine", "transform"):
            child = getattr(value, attribute, None)
            if child is not None:
                yield from visit(child, f"{path}.{attribute}")

    yield from visit(transform, "root")


def capture_transform_rng_state(transform: Any) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "class": type(value).__name__,
            "state": _numpy_rng_state_to_dict(rng.get_state()),
        }
        for path, value, rng in _iter_transform_random_states(transform)
    ]


def restore_transform_rng_state(
    transform: Any, saved: Sequence[Mapping[str, Any]]
) -> None:
    current = {
        path: (value, rng)
        for path, value, rng in _iter_transform_random_states(transform)
    }
    saved_paths = {str(item["path"]) for item in saved}
    if saved_paths != set(current):
        raise ValueError(
            "Checkpoint transform RNG layout does not match the current training "
            "transform pipeline. Use --resume-mode weights-only."
        )
    for item in saved:
        path = str(item["path"])
        value, rng = current[path]
        if item.get("class") != type(value).__name__:
            raise ValueError(
                f"Checkpoint transform at {path} is {item.get('class')!r}, but the "
                f"current transform is {type(value).__name__!r}."
            )
        rng.set_state(_numpy_rng_state_from_dict(item["state"]))


def capture_rng_state(
    generator: torch.Generator, transform: Any | None = None
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        # Store NumPy state as builtins so the complete checkpoint remains
        # readable by torch.load(weights_only=True).
        "numpy": _numpy_rng_state_to_dict(np.random.get_state()),
        "torch": torch.get_rng_state(),
        "dataloader_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if transform is not None:
        state["transform"] = capture_transform_rng_state(transform)
    return state


def restore_rng_state(
    state: Mapping[str, Any],
    generator: torch.Generator,
    transform: Any | None = None,
) -> None:
    """Restore every RNG present in a full checkpoint, tolerating older files."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        numpy_state = state["numpy"]
        if isinstance(numpy_state, Mapping):
            np.random.set_state(_numpy_rng_state_from_dict(numpy_state))
        else:
            # Backward compatibility with checkpoints created before RNG
            # metadata became weights-only-safe.
            np.random.set_state(numpy_state)
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "dataloader_generator" in state:
        generator.set_state(state["dataloader_generator"].cpu())
    if transform is not None:
        if "transform" in state:
            restore_transform_rng_state(transform, state["transform"])
        else:
            warnings.warn(
                "Checkpoint predates transform RNG capture; the resumed augmentation "
                "sequence may differ when num_workers=0.",
                RuntimeWarning,
                stacklevel=2,
            )


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Write a PyTorch artifact without exposing a partially written target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    scaler: Any,
    generator: torch.Generator,
    train_transform: Any | None = None,
    *,
    epoch: int,
    iteration: int,
    best_dice: float,
    best_train_dice: float,
    args: argparse.Namespace,
    model_config: H2ASegModelConfig,
    resolved_cases: Sequence[Mapping[str, str]],
    resolved_split: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    serialized_config = _json_safe_args(args)
    missing_config = sorted(set(RESUME_CONFIG_KEYS).difference(serialized_config))
    if missing_config:
        raise ValueError(
            "Cannot save a versioned full checkpoint without all resume "
            "configuration fields; missing: " + ", ".join(missing_config)
        )
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "artifact_type": CHECKPOINT_ARTIFACT_TYPE,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "amp_enabled": bool(scaler.is_enabled()),
        "epoch": epoch,
        "iteration": iteration,
        "best_dice": best_dice,
        "best_train_dice": best_train_dice,
        "validation_metric_policy": VALIDATION_METRIC_POLICY,
        "model_config": model_config.to_dict(),
        "config": serialized_config,
        "resolved_cases": _copy_cases(resolved_cases),
        "resolved_split": {
            name: _copy_cases(partition)
            for name, partition in resolved_split.items()
        },
        "rng_state": capture_rng_state(generator, train_transform),
    }
    _validate_full_checkpoint_payload(payload, versioned=True)
    atomic_torch_save(payload, path)


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location=device)


def _normalize_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Strip repeated wrapper prefixes without silently overwriting collisions."""
    if not state_dict:
        raise ValueError("Checkpoint model state_dict must not be empty.")
    prefixes = ("module.", "model.", "_model.")
    normalized: dict[str, Any] = {}
    origins: dict[str, str] = {}
    for original_key, value in state_dict.items():
        if not isinstance(original_key, str):
            raise ValueError("Checkpoint model state_dict keys must be strings.")
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break
        if not key:
            raise ValueError(
                f"Checkpoint key {original_key!r} becomes empty after prefix removal."
            )
        if key in normalized:
            raise ValueError(
                "Checkpoint key collision after wrapper-prefix removal: "
                f"{origins[key]!r} and {original_key!r} both map to {key!r}."
            )
        normalized[key] = value
        origins[key] = original_key
    return normalized


def _validate_full_checkpoint_payload(
    checkpoint: Mapping[str, Any], *, versioned: bool
) -> tuple[H2ASegModelConfig, Mapping[str, Any], bool]:
    """Validate full-resume state before mutating the model or optimizer."""
    for key in (
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "rng_state",
    ):
        if not isinstance(checkpoint[key], Mapping):
            raise ValueError(f"Checkpoint {key} must be a mapping.")

    for key in ("epoch", "iteration"):
        value = checkpoint[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"Checkpoint {key} must be a non-negative integer.")
    for key in ("best_dice", "best_train_dice"):
        value = checkpoint[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Checkpoint {key} must be a finite number in [0, 1].")
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Checkpoint {key} must be a finite number in [0, 1].")

    try:
        saved_model_config = H2ASegModelConfig.from_dict(checkpoint["model_config"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint contains invalid model_config metadata: {exc}") from exc

    saved_training_config = checkpoint["config"]
    if not isinstance(saved_training_config, Mapping):
        raise ValueError("Checkpoint config metadata must be a mapping.")
    if versioned:
        missing_config = sorted(
            set(RESUME_CONFIG_KEYS).difference(saved_training_config)
        )
        if missing_config:
            raise ValueError(
                "Versioned full checkpoint config is missing resume fields: "
                + ", ".join(missing_config)
            )

    resolved_cases = checkpoint["resolved_cases"]
    if (
        not isinstance(resolved_cases, Sequence)
        or isinstance(resolved_cases, (str, bytes))
        or not all(isinstance(case, Mapping) for case in resolved_cases)
    ):
        raise ValueError("Checkpoint resolved_cases must be a sequence of mappings.")
    resolved_split = checkpoint["resolved_split"]
    if not isinstance(resolved_split, Mapping):
        raise ValueError("Checkpoint resolved_split metadata must be a mapping.")
    for name in ("train", "val", "test"):
        partition = resolved_split.get(name)
        if (
            not isinstance(partition, Sequence)
            or isinstance(partition, (str, bytes))
            or not all(isinstance(case, Mapping) for case in partition)
        ):
            raise ValueError(
                f"Checkpoint resolved_split[{name!r}] must be a sequence of mappings."
            )

    saved_amp_enabled = checkpoint.get("amp_enabled")
    if saved_amp_enabled is None:
        if versioned:
            raise ValueError(
                "Versioned checkpoint amp_enabled metadata must be a boolean."
            )
        saved_amp_enabled = bool(checkpoint["scaler_state_dict"])
    if not isinstance(saved_amp_enabled, bool):
        raise ValueError("Checkpoint amp_enabled metadata must be a boolean.")
    saved_policy = checkpoint.get("validation_metric_policy")
    if versioned and (
        not isinstance(saved_policy, str) or not saved_policy.strip()
    ):
        raise ValueError(
            "Versioned checkpoint validation_metric_policy must be a non-empty string."
        )
    if not versioned and saved_policy is not None and not isinstance(saved_policy, str):
        raise ValueError("Checkpoint validation_metric_policy must be a string.")
    return saved_model_config, saved_training_config, saved_amp_enabled


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: StepLR | None = None,
    scaler: Any | None = None,
    generator: torch.Generator | None = None,
    expected_model_config: H2ASegModelConfig | None = None,
    train_transform: Any | None = None,
    expected_training_config: Mapping[str, Any] | None = None,
    expected_resolved_split: Mapping[
        str, Sequence[Mapping[str, str]]
    ] | None = None,
    amp_enabled: bool | None = None,
    resume_mode: str = "auto",
) -> dict[str, Any]:
    """Load full checkpoints and historical raw/wrapped state dictionaries."""
    if resume_mode not in {"auto", "full", "weights-only"}:
        raise ValueError(f"Unsupported resume_mode: {resume_mode!r}")
    # Deserialize the payload on CPU so full optimizer/scaler state does not
    # transiently occupy accelerator memory. load_state_dict moves tensors to
    # their owning parameter devices.
    checkpoint = _torch_load(path, torch.device("cpu"))
    if (
        isinstance(checkpoint, Mapping)
        and "checkpoint_format_version" in checkpoint
        and "model_state_dict" not in checkpoint
    ):
        raise ValueError(
            "Versioned full checkpoint is missing required field: model_state_dict"
        )
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        if not isinstance(model_state_dict, Mapping):
            raise ValueError("Checkpoint model_state_dict must be a mapping.")
        versioned = "checkpoint_format_version" in checkpoint
        version = checkpoint.get("checkpoint_format_version")
        if versioned and (
            type(version) is not int
            or version < 1
            or version > CHECKPOINT_FORMAT_VERSION
        ):
            raise ValueError(
                f"Unsupported checkpoint_format_version={version!r}; this code "
                f"supports versions up to {CHECKPOINT_FORMAT_VERSION}."
            )
        required_full_keys = (
            VERSIONED_FULL_RESUME_KEYS if versioned else FULL_RESUME_KEYS
        )
        missing_full_keys = sorted(required_full_keys.difference(checkpoint))
        if versioned and missing_full_keys:
            raise ValueError(
                "Versioned full checkpoint is missing required fields: "
                + ", ".join(missing_full_keys)
            )
        artifact_type = checkpoint.get("artifact_type")
        if (
            (versioned and artifact_type != CHECKPOINT_ARTIFACT_TYPE)
            or (
                not versioned
                and artifact_type is not None
                and artifact_type != CHECKPOINT_ARTIFACT_TYPE
            )
        ):
            raise ValueError(f"Unsupported checkpoint artifact_type={artifact_type!r}")

        is_full_checkpoint = not missing_full_keys
        if resume_mode == "full" and not is_full_checkpoint:
            raise ValueError(
                "Checkpoint is not a complete full-training checkpoint; missing: "
                + ", ".join(missing_full_keys)
            )
        if (
            resume_mode == "auto"
            and not is_full_checkpoint
            and TRAINING_STATE_HINT_KEYS.intersection(checkpoint)
        ):
            raise ValueError(
                "Checkpoint contains partial training metadata but is not complete; "
                "refusing to silently treat it as weights-only. Re-run with "
                "--resume-mode weights-only if discarding training state is intended. "
                f"Missing: {', '.join(missing_full_keys)}"
            )

        saved_model_config: H2ASegModelConfig | None = None
        saved_training_config: Mapping[str, Any] | None = None
        saved_amp_enabled: bool | None = None
        if is_full_checkpoint:
            (
                saved_model_config,
                saved_training_config,
                saved_amp_enabled,
            ) = _validate_full_checkpoint_payload(
                checkpoint, versioned=versioned
            )
        elif "model_config" in checkpoint and expected_model_config is not None:
            saved_model_config = H2ASegModelConfig.from_dict(
                checkpoint["model_config"]
            )

        if expected_model_config is not None and saved_model_config is not None:
            if saved_model_config != expected_model_config:
                raise ValueError(
                    "Checkpoint model_config does not match this training configuration: "
                    f"saved={saved_model_config.to_dict()}, "
                    f"expected={expected_model_config.to_dict()}"
                )

        restore_full_state = is_full_checkpoint and resume_mode != "weights-only"
        if restore_full_state:
            if not versioned:
                missing_new_fields = []
                if checkpoint.get("artifact_type") is None:
                    missing_new_fields.append("artifact_type")
                if not isinstance(checkpoint.get("amp_enabled"), bool):
                    missing_new_fields.append("amp_enabled")
                policy = checkpoint.get("validation_metric_policy")
                if not isinstance(policy, str) or not policy.strip():
                    missing_new_fields.append("validation_metric_policy")
                missing_legacy_config = sorted(
                    set(RESUME_CONFIG_KEYS).difference(saved_training_config or {})
                )
                if missing_new_fields or missing_legacy_config:
                    details = missing_new_fields + [
                        f"config.{key}" for key in missing_legacy_config
                    ]
                    warnings.warn(
                        "Loading a versionless previous full checkpoint; missing "
                        "new metadata will use compatibility behavior: "
                        + ", ".join(details),
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if expected_resolved_split is not None:
                validate_resume_split(
                    checkpoint["resolved_split"], expected_resolved_split
                )
            if expected_training_config is not None:
                validate_resume_config(
                    saved_training_config or {}, expected_training_config
                )
            if amp_enabled is not None:
                if saved_amp_enabled != amp_enabled:
                    raise ValueError(
                        "Full resume cannot change AMP mode: "
                        f"checkpoint={saved_amp_enabled}, current={amp_enabled}. "
                        "Use matching --amp/device settings or --resume-mode weights-only."
                    )

        model.load_state_dict(_normalize_state_dict(model_state_dict))
        if not restore_full_state:
            return {}
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if generator is not None:
            restore_rng_state(checkpoint["rng_state"], generator, train_transform)
        # Do not hand the caller a second long-lived reference to model and
        # Adam tensors after they have been restored into their owners.
        return {
            key: checkpoint[key]
            for key in RESUME_METADATA_KEYS
            if key in checkpoint
        }

    if resume_mode == "full":
        raise ValueError(
            "A raw/wrapped state_dict cannot be used for full resume; use "
            "--resume-mode weights-only or auto."
        )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint does not contain a model state_dict: {path}")
    wrapper_key = None
    state_dict: Mapping[str, Any] = checkpoint
    for key in ("state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            wrapper_key = key
            state_dict = candidate
            break
    if (
        resume_mode == "auto"
        and wrapper_key is not None
        and TRAINING_STATE_HINT_KEYS.intersection(checkpoint)
    ):
        raise ValueError(
            f"Checkpoint wrapper {wrapper_key!r} contains partial training metadata; "
            "use --resume-mode weights-only if discarding it is intended."
        )
    model.load_state_dict(_normalize_state_dict(state_dict))
    return {}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if min(
        args.batch_size,
        args.epochs,
        args.save_interval,
        args.val_interval,
        args.val_sw_batch_size,
    ) <= 0 or args.num_workers < 0:
        raise ValueError("Batch size, epochs and intervals must be positive; num-workers must be non-negative.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if not 0 <= args.val_overlap < 1:
        raise ValueError("--val-overlap must be in [0, 1).")

    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.num_workers))
    device = select_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if args.amp and not use_amp:
        print("AMP requested but CUDA is unavailable; continuing without AMP.")

    cases = discover_cases(
        args.ct_dir,
        args.pet_dir,
        args.label_dir,
        manifest=args.manifest,
        require_labels=True,
    )
    train_files, val_files, test_files = split_cases(
        cases, mode=args.split_mode, seed=args.split_seed
    )
    resolved_split = resolved_split_snapshot(train_files, val_files, test_files)
    output_dir = args.output_dir or (
        Path("save/hecktor2022_H2ASeg")
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Cases: total={len(cases)}, training={len(train_files)}, "
        f"validation={len(val_files)}, test={len(test_files)}; device={device}"
    )

    model_config = DEFAULT_MODEL_CONFIG
    train_transforms, val_transforms = build_transforms(model_config.patch_size)
    train_ds = monai.data.Dataset(train_files, transform=train_transforms)
    val_ds = monai.data.Dataset(val_files, transform=val_transforms)
    loader_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        pin_memory=device.type == "cuda",
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    # Full volumes can differ in shape and are validated one case at a time.
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kwargs)

    model = H2ASeg(**model_config.to_model_kwargs()).to(device)
    criterion_dice = monai.losses.DiceLoss(
        include_background=True, to_onehot_y=True, softmax=True
    ).to(device)
    criterion_ce = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.lr / 10)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    scaler = create_grad_scaler(use_amp)

    start_epoch = iteration = 0
    best_dice = best_train_dice = 0.0
    if args.resume:
        metadata = load_checkpoint(
            args.resume,
            model,
            device,
            optimizer,
            scheduler,
            scaler,
            generator,
            train_transform=train_transforms if args.num_workers == 0 else None,
            expected_model_config=model_config,
            expected_training_config=resume_training_config(args),
            expected_resolved_split=resolved_split,
            amp_enabled=use_amp,
            resume_mode=args.resume_mode,
        )
        start_epoch = int(metadata.get("epoch", -1)) + 1
        iteration = int(metadata.get("iteration", 0))
        best_dice = resolve_resumed_best_dice(metadata)
        best_train_dice = float(metadata.get("best_train_dice", 0.0))
        if metadata:
            effective_lrs = [group["lr"] for group in optimizer.param_groups]
            print(
                f"Loaded full state from {args.resume}; "
                f"starting at epoch {start_epoch + 1}; "
                f"restored optimizer LR(s)={effective_lrs}."
            )
        else:
            print(
                f"Loaded model weights only from {args.resume}; optimizer, scheduler, "
                "and epoch start fresh."
            )

    # Only publish/replace this run artifact after a full checkpoint's split
    # and trajectory metadata have passed validation.
    write_resolved_split(output_dir / "resolved_split.csv", resolved_split)

    writer = SummaryWriter(str(output_dir / "logs"))
    try:
        for epoch in range(start_epoch, args.epochs):
            started = time.time()
            model.train()
            epoch_metrics = []
            for step, batch in enumerate(train_loader):
                iteration += 1
                ct = batch["ct"].to(device=device, dtype=torch.float32, non_blocking=True)
                pet = batch["pet"].to(device=device, dtype=torch.float32, non_blocking=True)
                labels = binarize_labels(batch["label"], device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(ct, pet)
                    if len(outputs) != len(DEFAULT_LOSS_WEIGHTS):
                        raise RuntimeError(
                            "H2ASeg deep supervision contract changed: expected "
                            f"{len(DEFAULT_LOSS_WEIGHTS)} outputs, got {len(outputs)}."
                        )
                    losses = [
                        criterion_ce(output, labels.squeeze(1))
                        + criterion_dice(input=output, target=labels)
                        for output in outputs
                    ]
                    loss = sum(
                        weight * value
                        for weight, value in zip(DEFAULT_LOSS_WEIGHTS, losses)
                    )
                ensure_finite_losses(loss, losses)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                values = [value.item() for value in losses]
                fp, fn, iou, dice = show_deep_metrics(outputs, labels, True)
                row = [loss.item(), fp, fn, iou, dice]
                epoch_metrics.append(row)
                print(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S} {epoch + 1}/{args.epochs} "
                    f"{step + 1}/{len(train_loader)} loss={row[0]:.4f} dice={dice:.4f}"
                )
                writer.add_scalar("Training Loss", row[0], iteration)
                for index, value in enumerate(values, 1):
                    writer.add_scalar(f"Training Loss{index}", value, iteration)
                for name, value in zip(("FP", "FN", "IoU", "Dice"), (fp, fn, iou, dice)):
                    writer.add_scalar(f"Training {name}", value, iteration)
            scheduler.step()
            mean_train = np.mean(epoch_metrics, axis=0)
            if mean_train[4] >= best_train_dice:
                best_train_dice = float(mean_train[4])
                atomic_torch_save(model.state_dict(), output_dir / "train_best.pth")

            if should_validate_epoch(epoch, args.epochs, args.val_interval):
                model.eval()
                validation_cases: list[tuple[bool, Sequence[float]]] = []
                with torch.inference_mode():
                    for batch in val_loader:
                        # Keep complete source volumes and stitched logits on CPU;
                        # sliding_window_inference moves only each patch to the
                        # model device via sw_device.
                        ct = batch["ct"].to(dtype=torch.float32)
                        pet = batch["pet"].to(dtype=torch.float32)
                        labels = binarize_labels(batch["label"], torch.device("cpu"))
                        with torch.autocast(device_type=device.type, enabled=use_amp):
                            logits = predict_validation_logits(
                                model,
                                ct,
                                pet,
                                roi_size=model_config.patch_size,
                                sw_batch_size=args.val_sw_batch_size,
                                overlap=args.val_overlap,
                                model_device=device,
                            )
                            prediction = logits.argmax(dim=1, keepdim=True)
                        validation_cases.append(
                            (
                                bool(torch.any(labels).item()),
                                metrics_tensor(labels, prediction),
                            )
                        )
                validation_summary = summarize_validation_cases(validation_cases)
                mean_val = validation_summary.positive_mean
                if mean_val[3] >= best_dice:
                    best_dice = float(mean_val[3])
                    atomic_torch_save(model.state_dict(), output_dir / "val_best.pth")
                for name, value in zip(("FPR", "FNR", "IoU", "Dice"), mean_val):
                    writer.add_scalar(
                        f"Validation Positive-GT/{name}", value, epoch + 1
                    )
                writer.add_scalar(
                    "Validation Cohorts/Positive Cases",
                    validation_summary.positive_cases,
                    epoch + 1,
                )
                writer.add_scalar(
                    "Validation Cohorts/Negative Cases",
                    validation_summary.negative_cases,
                    epoch + 1,
                )
                negative_summary = "negative-GT n=0"
                if validation_summary.negative_mean_fpr is not None:
                    writer.add_scalar(
                        "Validation Negative-GT/Mean FPR",
                        validation_summary.negative_mean_fpr,
                        epoch + 1,
                    )
                    negative_summary = (
                        f"negative-GT n={validation_summary.negative_cases} "
                        f"mean FPR={validation_summary.negative_mean_fpr:.4f}"
                    )
                print(
                    f"Validation positive-GT n={validation_summary.positive_cases}: "
                    f"FPR={mean_val[0]:.4f} FNR={mean_val[1]:.4f} "
                    f"IoU={mean_val[2]:.4f} Dice={mean_val[3]:.4f} "
                    f"(best={best_dice:.4f}); {negative_summary}"
                )

            if (epoch + 1) % args.save_interval == 0:
                # Keep the historical model-only artifact for old consumers.
                atomic_torch_save(model.state_dict(), output_dir / f"{epoch}.pth")
            save_checkpoint(
                output_dir / "last.ckpt",
                model,
                optimizer,
                scheduler,
                scaler,
                generator,
                train_transforms if args.num_workers == 0 else None,
                epoch=epoch,
                iteration=iteration,
                best_dice=best_dice,
                best_train_dice=best_train_dice,
                args=args,
                model_config=model_config,
                resolved_cases=cases,
                resolved_split=resolved_split,
            )
            print(
                f"Epoch {epoch + 1} finished in {time.time() - started:.1f}s; "
                f"train dice={mean_train[4]:.4f}"
            )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
