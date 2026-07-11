"""Command-line inference and evaluation for H2ASeg.

The entrypoint accepts both the original raw ``state_dict`` checkpoints and
full training checkpoints.  CT and PET files are paired by case identifier,
never by their position in a sorted directory listing.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from model.H2ASeg import H2ASeg
from utils.data import ValidateGeometryd, discover_cases
from utils.metrics import metrics_tensor
from utils.model_config import DEFAULT_MODEL_CONFIG, H2ASegModelConfig


# Backward-compatible public name; discovery itself has a single canonical
# implementation in utils.data.
pair_cases = discover_cases


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({value}), but CUDA is unavailable")
    return device


def _load_checkpoint_file(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=device)


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a state_dict or a mapping containing one")
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            checkpoint = candidate
            break
    if not checkpoint or not all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        raise ValueError("Could not find a tensor model state_dict in checkpoint")
    return checkpoint  # type: ignore[return-value]


def _extract_model_config(checkpoint: object) -> H2ASegModelConfig | None:
    """Return validated model metadata from a full checkpoint, if present."""
    if not isinstance(checkpoint, Mapping) or "model_config" not in checkpoint:
        return None
    try:
        return H2ASegModelConfig.from_dict(checkpoint["model_config"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint contains invalid model_config metadata: {exc}") from exc


def resolve_model_config(
    checkpoint: object,
    *,
    roi_size: Sequence[int] | None = None,
    num_channels: Sequence[int] | None = None,
    window_size: Sequence[int] | None = None,
) -> H2ASegModelConfig:
    """Resolve CLI values against authoritative full-checkpoint metadata."""
    stored = _extract_model_config(checkpoint)
    overrides = {
        "patch_size": tuple(roi_size) if roi_size is not None else None,
        "num_channels": tuple(num_channels) if num_channels is not None else None,
        "window_size": tuple(window_size) if window_size is not None else None,
    }
    if stored is not None:
        candidate_values = stored.to_dict()
        candidate_values.update(
            {name: value for name, value in overrides.items() if value is not None}
        )
        candidate = H2ASegModelConfig.from_dict(candidate_values)
        conflicts = [
            f"{name}: checkpoint={getattr(stored, name)}, CLI={getattr(candidate, name)}"
            for name, value in overrides.items()
            if value is not None and getattr(candidate, name) != getattr(stored, name)
        ]
        if conflicts:
            raise ValueError(
                "Explicit model configuration conflicts with checkpoint metadata; "
                + "; ".join(conflicts)
            )
        return stored

    values = DEFAULT_MODEL_CONFIG.to_dict()
    values.update({name: value for name, value in overrides.items() if value is not None})
    return H2ASegModelConfig.from_dict(values)


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    *,
    checkpoint: object | None = None,
) -> None:
    """Load legacy/full checkpoints and common wrapper-prefixed state dicts."""
    if checkpoint is None:
        checkpoint = _load_checkpoint_file(path, device)
    state = _extract_state_dict(checkpoint)
    prefixes = ("module.", "_model.", "model.")
    normalized = {}
    for key, value in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        normalized[key] = value
    model.load_state_dict(normalized, strict=True)


class InferencePredictor(torch.nn.Module):
    """Adapt concatenated MONAI windows to H2ASeg's two-input interface."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs[:, :1], inputs[:, 1:2])
        if isinstance(outputs, (tuple, list)):
            if not outputs:
                raise RuntimeError("Model returned no deep-supervision outputs")
            return outputs[0]
        return outputs


def binary_metrics(label: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    """Return the repository's canonical binary segmentation metrics."""
    false_positive_rate, false_negative_rate, iou, dice = metrics_tensor(
        label, prediction
    )
    return {
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "iou": iou,
        "dice": dice,
    }


def hausdorff95(
    label: torch.Tensor, prediction: torch.Tensor, spacing: Sequence[float]
) -> float:
    """Return spacing-aware HD95, or NaN when either surface is empty."""
    if not torch.any(label) or not torch.any(prediction):
        return float("nan")
    from monai.metrics import compute_hausdorff_distance

    value = compute_hausdorff_distance(
        prediction.bool(),
        label.bool(),
        include_background=True,
        percentile=95,
        spacing=tuple(spacing),
    )
    return float(value.item())


def save_prediction(prediction: torch.Tensor, reference_path: Path, output_path: Path) -> None:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("Saving predictions requires nibabel") from exc
    reference = nib.load(str(reference_path))
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(
        prediction.squeeze().cpu().numpy().astype(np.uint8),
        reference.affine,
        header,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))


def build_inference_transform(keys: Sequence[str]) -> Compose:
    """Load inputs and reject spatially misaligned multimodal cases."""
    keys = tuple(keys)
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ValidateGeometryd(keys=keys, reference_key="ct"),
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run H2ASeg PET/CT inference")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        type=Path,
        help="CSV with case, ct, pet and optional label columns",
    )
    source.add_argument("--ct-dir", type=Path, help="CT directory (requires --pet-dir)")
    parser.add_argument("--pet-dir", type=Path, help="PET directory (requires --ct-dir)")
    parser.add_argument("--label-dir", type=Path, help="Optional labels for evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, default=Path("inference_metrics.csv"))
    parser.add_argument("--prediction-dir", type=Path, help="Save NIfTI masks here")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--roi-size", type=int, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Sliding-window/model patch size (default: checkpoint metadata or MICCAI 2024 config)",
    )
    parser.add_argument("--sw-batch-size", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--num-channels", type=int, nargs=5, default=None,
        metavar=("C1", "C2", "C3", "C4", "C5"),
        help="Encoder channels (default: checkpoint metadata or MICCAI 2024 config)",
    )
    parser.add_argument(
        "--window-size", type=int, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Attention window (default: checkpoint metadata or MICCAI 2024 config)",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0, 1)")
    if args.sw_batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers non-negative")
    device = resolve_device(args.device)
    records = discover_cases(
        args.ct_dir,
        args.pet_dir,
        args.label_dir,
        manifest=args.manifest,
    )
    has_labels = "label" in records[0]
    keys = ["ct", "pet"] + (["label"] if has_labels else [])
    dataset = Dataset(records, build_inference_transform(keys))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = _load_checkpoint_file(args.checkpoint, device)
    model_config = resolve_model_config(
        checkpoint,
        roi_size=args.roi_size,
        num_channels=args.num_channels,
        window_size=args.window_size,
    )
    model = H2ASeg(**model_config.to_model_kwargs()).to(device)
    load_checkpoint(model, args.checkpoint, device, checkpoint=checkpoint)
    predictor = InferencePredictor(model).eval()
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            case = batch["case"][0]
            ct = batch["ct"].float().to(device, non_blocking=True)
            pet = batch["pet"].float().to(device, non_blocking=True)
            logits = sliding_window_inference(
                torch.cat((ct, pet), dim=1),
                roi_size=model_config.patch_size,
                sw_batch_size=args.sw_batch_size,
                predictor=predictor,
                overlap=args.overlap,
            )
            prediction = logits.argmax(dim=1, keepdim=True)
            row: dict[str, object] = {
                "case": case,
                "prediction_voxels": int(prediction.sum().item()),
            }
            if has_labels:
                label = (batch["label"].to(device, non_blocking=True) != 0)
                row.update(binary_metrics(label, prediction))
                affine = batch["ct"].affine.to(device)
                if affine.ndim == 3:
                    affine = affine[0]
                spacing = torch.linalg.vector_norm(affine[:3, :3], dim=0).tolist()
                row["hausdorff95"] = hausdorff95(label, prediction, spacing)
                row["label_voxels"] = int(label.sum().item())
            rows.append(row)
            if args.prediction_dir:
                save_prediction(
                    prediction[0, 0],
                    Path(batch["ct"].meta["filename_or_obj"][0]),
                    args.prediction_dir / f"{case}.nii.gz",
                )
            print(f"[{len(rows)}/{len(records)}] {case}")

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} case(s) to {args.metrics_csv}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
