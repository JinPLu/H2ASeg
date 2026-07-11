"""Train H2ASeg on aligned PET/CT NIfTI volumes.

The MICCAI 2024 model, losses, and default hyperparameters are unchanged.  This
entry point adds explicit data contracts, reproducible splits, and resumable
training state around the original implementation.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
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
FULL_RESUME_KEYS = frozenset(
    {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "epoch",
        "iteration",
        "model_config",
        "resolved_split",
        "rng_state",
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train H2ASeg on paired PET/CT NIfTI data.")
    parser.add_argument("--manifest", type=Path, help="CSV with case,ct,pet,label[,split] columns.")
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
    load_and_validate = [
        LoadImaged(keys=["pet", "ct", "label"]),
        EnsureChannelFirstd(keys=["pet", "ct", "label"]),
        ValidateGeometryd(
            keys=["ct", "pet", "label"], reference_key="ct", case_key="case"
        ),
    ]
    train = Compose(
        load_and_validate
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
    val = Compose(load_and_validate)
    return train, val


def _json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "ct", "pet", "label", "split"))
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            for case in resolved_split.get(split_name, ()):
                writer.writerow(
                    {
                        "case": case["case"],
                        "ct": case["ct"],
                        "pet": case["pet"],
                        "label": case["label"],
                        "split": split_name,
                    }
                )


def validate_resume_split(
    saved: Mapping[str, Sequence[Mapping[str, str]]],
    current: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    """Prevent a full-state resume from silently changing its data assignment."""
    normalized_saved = {
        name: _copy_cases(saved.get(name, ())) for name in ("train", "val", "test")
    }
    normalized_current = {
        name: _copy_cases(current.get(name, ())) for name in ("train", "val", "test")
    }
    if normalized_saved != normalized_current:
        raise ValueError(
            "The current resolved data split differs from the full checkpoint. "
            "Use the checkpoint's data/manifest for exact resume, or load a "
            "legacy model-only .pth for transfer learning."
        )


def capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        # Store NumPy state as builtins so the complete checkpoint remains
        # readable by torch.load(weights_only=True).
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "dataloader_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any], generator: torch.Generator) -> None:
    """Restore every RNG present in a full checkpoint, tolerating older files."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        numpy_state = state["numpy"]
        if isinstance(numpy_state, Mapping):
            np.random.set_state(
                (
                    str(numpy_state["bit_generator"]),
                    np.asarray(numpy_state["keys"], dtype=np.uint32),
                    int(numpy_state["position"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    scaler: Any,
    generator: torch.Generator,
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
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "iteration": iteration,
            "best_dice": best_dice,
            "best_train_dice": best_train_dice,
            "model_config": model_config.to_dict(),
            "config": _json_safe_args(args),
            "resolved_cases": _copy_cases(resolved_cases),
            "resolved_split": {
                name: _copy_cases(partition)
                for name, partition in resolved_split.items()
            },
            "rng_state": capture_rng_state(generator),
        },
        path,
    )


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location=device)


def _remove_uniform_prefix(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return dict(state_dict)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: StepLR | None = None,
    scaler: Any | None = None,
    generator: torch.Generator | None = None,
    expected_model_config: H2ASegModelConfig | None = None,
) -> dict[str, Any]:
    """Load full checkpoints and historical raw/wrapped state dictionaries."""
    checkpoint = _torch_load(path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        if "model_config" in checkpoint and expected_model_config is not None:
            saved_config = H2ASegModelConfig.from_dict(checkpoint["model_config"])
            if saved_config != expected_model_config:
                raise ValueError(
                    "Checkpoint model_config does not match this training configuration: "
                    f"saved={saved_config.to_dict()}, expected={expected_model_config.to_dict()}"
                )
        model.load_state_dict(
            _remove_uniform_prefix(checkpoint["model_state_dict"], "module.")
        )
        if not FULL_RESUME_KEYS.issubset(checkpoint):
            return {}
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if generator is not None and isinstance(checkpoint.get("rng_state"), Mapping):
            restore_rng_state(checkpoint["rng_state"], generator)
        return checkpoint

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint does not contain a model state_dict: {path}")
    model.load_state_dict(_remove_uniform_prefix(state_dict, "module."))
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
    write_resolved_split(output_dir / "resolved_split.csv", resolved_split)
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
            expected_model_config=model_config,
        )
        if isinstance(metadata.get("resolved_split"), Mapping):
            validate_resume_split(metadata["resolved_split"], resolved_split)
        start_epoch = int(metadata.get("epoch", -1)) + 1
        iteration = int(metadata.get("iteration", 0))
        best_dice = float(metadata.get("best_dice", 0.0))
        best_train_dice = float(metadata.get("best_train_dice", 0.0))
        if metadata:
            print(
                f"Loaded full state from {args.resume}; "
                f"starting at epoch {start_epoch + 1}."
            )
        else:
            print(
                f"Loaded model weights only from {args.resume}; optimizer, scheduler, "
                "and epoch start fresh."
            )

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
                labels = (
                    batch["label"].to(device=device, dtype=torch.long, non_blocking=True) != 0
                ).long()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(ct, pet)
                    losses = [
                        criterion_ce(output, labels.squeeze(1))
                        + criterion_dice(input=output, target=labels)
                        for output in outputs
                    ]
                    loss = sum(
                        weight * value
                        for weight, value in zip(DEFAULT_LOSS_WEIGHTS, losses)
                    )
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
                torch.save(model.state_dict(), output_dir / "train_best.pth")

            if (epoch + 1) % args.val_interval == 0:
                model.eval()
                validation = []
                with torch.inference_mode():
                    for batch in val_loader:
                        ct = batch["ct"].to(
                            device=device, dtype=torch.float32, non_blocking=True
                        )
                        pet = batch["pet"].to(
                            device=device, dtype=torch.float32, non_blocking=True
                        )
                        labels = (
                            batch["label"].to(
                                device=device, dtype=torch.long, non_blocking=True
                            )
                            != 0
                        ).long()
                        with torch.autocast(device_type=device.type, enabled=use_amp):
                            inputs = torch.cat((ct, pet), dim=1)

                            def predict_patch(patch: torch.Tensor) -> torch.Tensor:
                                return model(patch[:, :1], patch[:, 1:2])[0]

                            logits = sliding_window_inference(
                                inputs,
                                roi_size=model_config.patch_size,
                                sw_batch_size=args.val_sw_batch_size,
                                predictor=predict_patch,
                                overlap=args.val_overlap,
                            )
                            prediction = logits.argmax(dim=1, keepdim=True)
                        validation.append(metrics_tensor(labels, prediction))
                mean_val = np.mean(validation, axis=0)
                if mean_val[3] >= best_dice:
                    best_dice = float(mean_val[3])
                    torch.save(model.state_dict(), output_dir / "val_best.pth")
                for name, value in zip(("FP", "FN", "IoU", "Dice"), mean_val):
                    writer.add_scalar(f"Validating {name}", value, epoch + 1)
                print(
                    f"Validation: FP={mean_val[0]:.4f} FN={mean_val[1]:.4f} "
                    f"IoU={mean_val[2]:.4f} Dice={mean_val[3]:.4f} "
                    f"(best={best_dice:.4f})"
                )

            if (epoch + 1) % args.save_interval == 0:
                # Keep the historical model-only artifact for old consumers.
                torch.save(model.state_dict(), output_dir / f"{epoch}.pth")
            save_checkpoint(
                output_dir / "last.ckpt",
                model,
                optimizer,
                scheduler,
                scaler,
                generator,
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
