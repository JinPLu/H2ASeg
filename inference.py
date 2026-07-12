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


def _load_checkpoint_file(path: Path) -> object:
    """Load checkpoint payloads on CPU, independent of the model device."""
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=torch.device("cpu"))


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
    """Load legacy/full checkpoints and common wrapper-prefixed state dicts.

    Checkpoint payloads stay on CPU; ``load_state_dict`` copies only model
    parameters to the model's existing device.  ``device`` remains in the
    public signature for compatibility with existing callers.
    """
    if checkpoint is None:
        checkpoint = _load_checkpoint_file(path)
    state = _extract_state_dict(checkpoint)
    prefixes = ("module.", "_model.", "model.")
    normalized: dict[str, torch.Tensor] = {}
    origins: dict[str, str] = {}
    for key, value in state.items():
        original_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        if key in normalized:
            raise ValueError(
                "Checkpoint parameter keys collide after removing wrapper prefixes: "
                f"{origins[key]!r} and {original_key!r} both map to {key!r}"
            )
        normalized[key] = value
        origins[key] = original_key
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
    # metrics_tensor validates the shared [batch, channel, *spatial] contract
    # and treats channel 1 as foreground for two-channel one-hot masks.  Use
    # the same foreground view so all metrics in this result describe the
    # identical voxels.
    label_positive = label[:, 1:].bool() if label.shape[1] == 2 else label.bool()
    prediction_positive = (
        prediction[:, 1:].bool()
        if prediction.shape[1] == 2
        else prediction.bool()
    )
    true_positive = int(torch.count_nonzero(label_positive & prediction_positive).item())
    predicted_positive = int(torch.count_nonzero(prediction_positive).item())
    actual_positive = int(torch.count_nonzero(label_positive).item())
    precision = (
        true_positive / predicted_positive
        if predicted_positive
        else float("nan")
    )
    recall = true_positive / actual_positive if actual_positive else float("nan")
    return {
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
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
    if output_path.resolve(strict=False) == reference_path.resolve(strict=False):
        raise ValueError(f"Refusing to overwrite source image with prediction: {reference_path}")
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("Saving predictions requires nibabel") from exc
    if prediction.ndim != 3:
        raise ValueError(
            f"Prediction must be exactly 3D before export, got shape {tuple(prediction.shape)}"
        )
    reference = nib.load(str(reference_path))
    reference_shape = tuple(int(size) for size in reference.shape)
    prediction_shape = tuple(int(size) for size in prediction.shape)
    if len(reference_shape) != 3:
        raise ValueError(
            f"Reference image must be exactly 3D, got shape {reference_shape}: {reference_path}"
        )
    if prediction_shape != reference_shape:
        raise ValueError(
            "Prediction/reference spatial shape mismatch: "
            f"prediction={prediction_shape}, reference={reference_shape}"
        )
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(
        prediction.detach().cpu().numpy().astype(np.uint8),
        reference.affine,
        header,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))


def prediction_output_path(
    prediction_dir: Path,
    case: str,
    *,
    protected_paths: Sequence[Path] = (),
) -> Path:
    """Return a contained output path for one manifest case identifier."""
    if not isinstance(case, str) or not case:
        raise ValueError("Case identifier must be a non-empty string")
    if case in {".", ".."} or "/" in case or "\\" in case:
        raise ValueError(
            f"Unsafe case identifier {case!r}: path separators and dot segments are not allowed"
        )

    root = prediction_dir.resolve(strict=False)
    output = (root / f"{case}.nii.gz").resolve(strict=False)
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Unsafe case identifier {case!r}: prediction path escapes {root}"
        ) from exc
    protected = {path.resolve(strict=False) for path in protected_paths}
    if output in protected:
        raise ValueError(f"Refusing to overwrite case source image with prediction: {output}")
    return output


def prepare_prediction_output_paths(
    records: Sequence[Mapping[str, object]], prediction_dir: Path
) -> dict[str, Path]:
    """Resolve every destination and reject collisions with any case input."""
    protected_paths = {
        Path(str(record[key])).resolve(strict=False)
        for record in records
        for key in ("ct", "pet", "label")
        if record.get(key)
    }
    outputs: dict[str, Path] = {}
    for record in records:
        case = record.get("case")
        if not isinstance(case, str):
            raise ValueError("Every inference record must have a string case identifier")
        output = prediction_output_path(prediction_dir, case)
        if output in protected_paths:
            raise ValueError(
                f"Refusing to overwrite case source image with prediction: {output}"
            )
        outputs[case] = output
    return outputs


def sliding_window_logits(
    inputs: torch.Tensor,
    *,
    predictor: torch.nn.Module,
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: float,
    compute_device: torch.device,
) -> torch.Tensor:
    """Run windows on ``compute_device`` while stitching the full output on CPU."""
    if inputs.device.type != "cpu":
        raise ValueError(
            f"Full-volume sliding-window inputs must stay on CPU, got {inputs.device}"
        )
    logits = sliding_window_inference(
        inputs,
        roi_size=tuple(roi_size),
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=overlap,
        sw_device=compute_device,
        device=torch.device("cpu"),
    )
    if logits.device.type != "cpu":
        raise RuntimeError(
            f"Sliding-window output must be stitched on CPU, got {logits.device}"
        )
    return logits


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
    prediction_paths = (
        prepare_prediction_output_paths(records, args.prediction_dir)
        if args.prediction_dir
        else {}
    )
    dataset = Dataset(records, build_inference_transform(keys))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = _load_checkpoint_file(args.checkpoint)
    model_config = resolve_model_config(
        checkpoint,
        roi_size=args.roi_size,
        num_channels=args.num_channels,
        window_size=args.window_size,
    )
    model = H2ASeg(**model_config.to_model_kwargs()).to(device)
    load_checkpoint(model, args.checkpoint, device, checkpoint=checkpoint)
    del checkpoint
    predictor = InferencePredictor(model).eval()
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            case = batch["case"][0]
            ct = batch["ct"].float()
            pet = batch["pet"].float()
            logits = sliding_window_logits(
                torch.cat((ct, pet), dim=1),
                predictor=predictor,
                roi_size=model_config.patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.overlap,
                compute_device=device,
            )
            prediction = logits.argmax(dim=1, keepdim=True)
            row: dict[str, object] = {
                "case": case,
                "prediction_voxels": int(prediction.sum().item()),
            }
            if has_labels:
                label = batch["label"] != 0
                row.update(binary_metrics(label, prediction))
                affine = batch["ct"].affine
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
                    prediction_paths[case],
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
