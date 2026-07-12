#!/usr/bin/env python3
"""Strict, fast runtime verification for H2ASeg servers.

This is intentionally not a training entry point.  It exercises one real-model
forward/backward step, raw/full checkpoint loading, prediction export, and a
multi-window inference pass.  Missing runtime dependencies are an error rather
than a skipped/green smoke test.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import random
import sys
import tempfile
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


class RuntimeDependencyError(RuntimeError):
    """Raised when the documented ML runtime is incomplete or broken."""


@lru_cache(maxsize=1)
def require_runtime() -> Mapping[str, Any]:
    """Import every dependency needed by the smoke path, reporting all failures."""
    modules: dict[str, Any] = {}
    failures: list[str] = []
    for name in (
        "torch",
        "monai",
        "numpy",
        "scipy",
        "einops",
        "nibabel",
        "tensorboard",
    ):
        try:
            modules[name] = importlib.import_module(name)
        except Exception as exc:  # import can fail for ABI/runtime reasons too
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        detail = "\n  - ".join(failures)
        raise RuntimeDependencyError(
            "H2ASeg runtime verification cannot start because required "
            f"dependencies are unavailable:\n  - {detail}\n"
            "Install the server-appropriate PyTorch build, then run "
            "`python -m pip install -r requirements.txt`."
        )
    return modules


def smoke_model_config() -> Any:
    """Return the smallest practical configuration that traverses the real model."""
    require_runtime()
    from utils.model_config import H2ASegModelConfig

    return H2ASegModelConfig(
        patch_size=(32, 32, 32),
        num_channels=(4, 4, 4, 4, 4),
        window_size=(8, 8, 4),
        attention_dropout=0.0,
        encoder_dropout=0.0,
        decoder_dropout=0.0,
    )


def resolve_device(specification: str) -> Any:
    torch = require_runtime()["torch"]
    if specification == "auto":
        specification = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(specification)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("--device must resolve to cpu, cuda, or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested ({specification}), but torch.cuda.is_available() is False"
        )
    return device


def resolve_amp(mode: str, device: Any) -> bool:
    if mode not in {"auto", "on", "off"}:
        raise ValueError("AMP mode must be auto, on, or off")
    if mode == "auto":
        return device.type == "cuda"
    enabled = mode == "on"
    if enabled and device.type != "cuda":
        raise ValueError("--amp on requires a CUDA device")
    return enabled


def _assert_finite(tensor: Any, description: str) -> None:
    torch = require_runtime()["torch"]
    if not bool(torch.isfinite(tensor).all().item()):
        raise AssertionError(f"{description} contains NaN or Inf")


def _state_dict_cpu(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _assert_state_equal(model: Any, expected: Mapping[str, Any], description: str) -> None:
    torch = require_runtime()["torch"]
    actual = model.state_dict()
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise AssertionError(
            f"{description} state keys differ; missing={missing[:5]}, extra={extra[:5]}"
        )
    mismatched = [
        name
        for name, value in actual.items()
        if not torch.equal(value.detach().cpu(), expected[name])
    ]
    if mismatched:
        raise AssertionError(
            f"{description} changed {len(mismatched)} tensor(s): {mismatched[:5]}"
        )


def _assert_nested_equal(actual: Any, expected: Any, description: str) -> None:
    """Compare optimizer/scheduler state without assuming a tensor device."""
    torch = require_runtime()["torch"]
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(
            actual.detach().cpu(), expected.detach().cpu()
        ):
            raise AssertionError(f"{description} tensor differs")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AssertionError(f"{description} mapping keys differ")
        for key, value in expected.items():
            _assert_nested_equal(actual[key], value, f"{description}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise AssertionError(f"{description} sequence differs")
        for index, value in enumerate(expected):
            _assert_nested_equal(actual[index], value, f"{description}[{index}]")
        return
    if actual != expected:
        raise AssertionError(
            f"{description} differs: restored={actual!r}, expected={expected!r}"
        )


def run_forward_backward(
    *,
    device: Any,
    use_amp: bool,
    seed: int = 12345,
    force_overflow_attempts: int = 0,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Run one genuine five-head CE+Dice optimization step."""
    runtime = require_runtime()
    torch = runtime["torch"]
    monai = runtime["monai"]

    if (
        isinstance(force_overflow_attempts, bool)
        or not isinstance(force_overflow_attempts, int)
        or not 0 <= force_overflow_attempts <= 7
    ):
        raise ValueError("force_overflow_attempts must be an integer in [0, 7]")
    if force_overflow_attempts and (not use_amp or device.type != "cuda"):
        raise ValueError(
            "force_overflow_attempts is test-only and requires AMP on CUDA"
        )

    from model.H2ASeg import H2ASeg
    from train import DEFAULT_LOSS_WEIGHTS, create_grad_scaler

    random.seed(seed)
    runtime["numpy"].random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = smoke_model_config()
    model = H2ASeg(**config.to_model_kwargs()).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = create_grad_scaler(use_amp)
    criterion_ce = torch.nn.CrossEntropyLoss().to(device)
    criterion_dice = monai.losses.DiceLoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
    ).to(device)

    ct = torch.randn(
        (1, 1, *config.patch_size),
        device=device,
        dtype=torch.float32,
    )
    pet = torch.randn_like(ct)
    labels = torch.zeros(
        (1, 1, *config.patch_size), device=device, dtype=torch.long
    )
    labels[
        :,
        :,
        config.patch_size[0] // 4 : config.patch_size[0] // 2,
        config.patch_size[1] // 4 : config.patch_size[1] // 2,
        config.patch_size[2] // 4 : config.patch_size[2] // 2,
    ] = 1

    retry_cpu_rng = torch.get_rng_state().clone()
    retry_cuda_rng = (
        torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    )
    max_attempts = 8 if use_amp else 1
    initial_scale = float(scaler.get_scale())
    overflow_events: list[dict[str, Any]] = []
    successful_attempt: int | None = None
    expected_shape = (1, 2, *config.patch_size)

    for attempt in range(1, max_attempts + 1):
        # An overflow retry must see the same batch and stochastic-model RNG.
        torch.set_rng_state(retry_cpu_rng)
        if retry_cuda_rng is not None:
            torch.cuda.set_rng_state(retry_cuda_rng, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(ct, pet)
            if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
                raise AssertionError(
                    "H2ASeg must return five deep-supervision heads, got "
                    f"{type(outputs)!r}"
                )
            output_shapes = [tuple(output.shape) for output in outputs]
            if any(shape != expected_shape for shape in output_shapes):
                raise AssertionError(
                    f"Unexpected output shapes: {output_shapes}; "
                    f"expected {expected_shape}"
                )
            head_losses = [
                criterion_ce(output, labels.squeeze(1))
                + criterion_dice(input=output, target=labels)
                for output in outputs
            ]
            if len(DEFAULT_LOSS_WEIGHTS) != len(head_losses):
                raise AssertionError(
                    "Deep-supervision output/loss-weight count mismatch: "
                    f"{len(head_losses)} heads versus "
                    f"{len(DEFAULT_LOSS_WEIGHTS)} weights"
                )
            total_loss = sum(
                weight * value
                for weight, value in zip(
                    DEFAULT_LOSS_WEIGHTS, head_losses, strict=True
                )
            )

        # Scale warm-down cannot repair a non-finite forward objective.
        for index, value in enumerate(head_losses, 1):
            _assert_finite(value, f"head {index} CE+Dice loss")
        _assert_finite(total_loss, "weighted deep-supervision loss")
        scale_before = float(scaler.get_scale())
        scaler.scale(total_loss).backward()
        forced_overflow = attempt <= force_overflow_attempts
        if forced_overflow:
            gradient_to_poison = next(
                (
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ),
                None,
            )
            if gradient_to_poison is None:
                raise AssertionError(
                    "Cannot force AMP overflow because no model gradient exists"
                )
            with torch.no_grad():
                gradient_to_poison.reshape(-1)[0] = float("inf")
        scaler.unscale_(optimizer)

        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        missing_gradients = sum(
            parameter.grad is None for parameter in trainable_parameters
        )
        if missing_gradients:
            raise AssertionError(
                f"{missing_gradients} trainable parameter(s) received no gradient"
            )
        gradients = [parameter.grad for parameter in trainable_parameters]
        bad_gradient_tensors = 0
        bad_gradient_elements = 0
        for gradient in gradients:
            finite = torch.isfinite(gradient)
            nonfinite_elements = int((~finite).sum().item())
            if nonfinite_elements:
                bad_gradient_tensors += 1
                bad_gradient_elements += nonfinite_elements

        if bad_gradient_tensors:
            if not use_amp:
                raise FloatingPointError(
                    "Non-finite gradients with AMP disabled: "
                    f"tensors={bad_gradient_tensors}, "
                    f"elements={bad_gradient_elements}"
                )
            parameters_before_skip = [
                parameter.detach().clone() for parameter in model.parameters()
            ]
            optimizer_entries_before_skip = len(optimizer.state)
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if any(
                not torch.equal(parameter.detach(), before)
                for parameter, before in zip(
                    model.parameters(), parameters_before_skip, strict=True
                )
            ) or len(optimizer.state) != optimizer_entries_before_skip:
                raise AssertionError(
                    "GradScaler did not skip the optimizer step after overflow"
                )
            if not scale_after < scale_before:
                raise AssertionError(
                    "GradScaler overflow retry requires a strictly smaller scale: "
                    f"before={scale_before}, after={scale_after}"
                )
            overflow_events.append(
                {
                    "attempt": attempt,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                    "bad_gradient_tensors": bad_gradient_tensors,
                    "bad_gradient_elements": bad_gradient_elements,
                    "forced": forced_overflow,
                    "optimizer_step_skipped": True,
                }
            )
            optimizer.zero_grad(set_to_none=True)
            del (
                outputs,
                head_losses,
                total_loss,
                gradients,
                trainable_parameters,
                parameters_before_skip,
            )
            continue

        if not any(
            bool(torch.count_nonzero(gradient).item()) for gradient in gradients
        ):
            raise AssertionError("Every model gradient is zero")

        heads_with_grad: list[int] = []
        for index in range(1, 6):
            head = getattr(model.decoder, f"output{index}")
            head_gradients = [
                parameter.grad
                for parameter in head.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not head_gradients or not any(
                bool(torch.count_nonzero(gradient).item())
                for gradient in head_gradients
            ):
                raise AssertionError(
                    f"Deep-supervision output head {index} received no "
                    "non-zero gradient"
                )
            heads_with_grad.append(index)

        parameters_before_step = [
            parameter.detach().clone() for parameter in model.parameters()
        ]
        scaler.step(optimizer)
        scaler.update()
        changed_parameters = sum(
            not torch.equal(parameter.detach(), before)
            for parameter, before in zip(
                model.parameters(), parameters_before_step, strict=True
            )
        )
        if not changed_parameters:
            raise AssertionError(
                "Finite gradients did not change any model parameter"
            )
        del parameters_before_step
        successful_attempt = attempt
        break

    if successful_attempt is None:
        raise FloatingPointError(
            f"AMP gradients remained non-finite after {max_attempts} attempts; "
            f"overflow_events={json.dumps(overflow_events, sort_keys=True)}"
        )

    final_scale = float(scaler.get_scale())
    result = {
        "output_shapes": output_shapes,
        "head_losses": [float(value.detach().cpu()) for value in head_losses],
        "weighted_loss": float(total_loss.detach().cpu()),
        "parameters_with_grad": len(gradients),
        "heads_with_grad": heads_with_grad,
        "amp": use_amp,
        "initial_grad_scale": initial_scale,
        "final_grad_scale": final_scale,
        "optimization_attempts": successful_attempt,
        "overflow_retries": len(overflow_events),
        "overflow_events": overflow_events,
        "changed_parameters": changed_parameters,
        "optimizer_step_verified": True,
    }
    optimizer.zero_grad(set_to_none=True)
    return model, config, optimizer, scaler, result


def _weights_only_load(path: Path) -> Any:
    torch = require_runtime()["torch"]
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_checkpoint_smoke(
    model: Any,
    config: Any,
    optimizer: Any,
    scaler: Any,
    *,
    device: Any,
    directory: Path,
) -> dict[str, Any]:
    """Round-trip real H2ASeg weights through raw and current full formats."""
    torch = require_runtime()["torch"]

    import inference
    import train
    from model.H2ASeg import H2ASeg

    expected = _state_dict_cpu(model)
    raw_path = directory / "raw_state_dict.pth"
    torch.save(expected, raw_path)

    inference_raw_target = H2ASeg(**config.to_model_kwargs()).to(device)
    inference.load_checkpoint(inference_raw_target, raw_path, device)
    _assert_state_equal(inference_raw_target, expected, "raw inference load")
    del inference_raw_target

    training_raw_target = H2ASeg(**config.to_model_kwargs()).to(device)
    training_metadata = train.load_checkpoint(
        raw_path,
        training_raw_target,
        device,
        expected_model_config=config,
    )
    if training_metadata:
        raise AssertionError("A raw state_dict must remain a weights-only checkpoint")
    _assert_state_equal(training_raw_target, expected, "raw training load")
    del training_raw_target

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    source_optimizer_state = optimizer.state_dict()
    if not source_optimizer_state["state"]:
        raise AssertionError(
            "Adam state is empty; checkpoint smoke requires a real optimizer step"
        )
    moment_tensor_count = sum(
        int("exp_avg" in state) + int("exp_avg_sq" in state)
        for state in source_optimizer_state["state"].values()
    )
    if moment_tensor_count == 0:
        raise AssertionError("Adam checkpoint has no first/second-moment tensors")
    source_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    scaler_enabled = bool(scaler.is_enabled())
    generator = torch.Generator().manual_seed(17)
    source_scheduler_state = scheduler.state_dict()
    source_scaler_state = scaler.state_dict()
    source_generator_state = generator.get_state().clone()
    cases = [
        {
            "case": "RUNTIME_SMOKE",
            "ct": "ct.nii.gz",
            "pet": "pet.nii.gz",
            "label": "label.nii.gz",
        }
    ]
    resolved_split = {name: cases for name in ("train", "val", "test")}
    args = argparse.Namespace(
        device=str(device),
        amp=scaler_enabled,
        lr=1e-4,
        batch_size=1,
        num_workers=0,
        seed=12345,
        split_mode="seeded",
        split_seed=12345,
        val_interval=1,
        val_sw_batch_size=1,
        val_overlap=0.25,
        epochs=1,
    )
    full_path = directory / "maintained_full.ckpt"
    train.save_checkpoint(
        full_path,
        model,
        optimizer,
        scheduler,
        scaler,
        generator,
        epoch=2,
        iteration=7,
        best_dice=0.25,
        best_train_dice=0.5,
        args=args,
        model_config=config,
        resolved_cases=cases,
        resolved_split=resolved_split,
    )

    payload = _weights_only_load(full_path)
    if payload.get("checkpoint_format_version") != train.CHECKPOINT_FORMAT_VERSION:
        raise AssertionError("Full checkpoint format version did not round-trip")
    if payload.get("artifact_type") != train.CHECKPOINT_ARTIFACT_TYPE:
        raise AssertionError("Full checkpoint artifact type did not round-trip")
    if payload.get("amp_enabled") is not scaler_enabled:
        raise AssertionError("Full checkpoint AMP metadata did not round-trip")
    if inference.resolve_model_config(payload) != config:
        raise AssertionError("Full checkpoint model_config did not round-trip")
    inference_target = H2ASeg(**config.to_model_kwargs()).to(device)
    inference.load_checkpoint(
        inference_target,
        full_path,
        device,
        checkpoint=payload,
    )
    _assert_state_equal(inference_target, expected, "full inference load")
    del inference_target

    resume_target = H2ASeg(**config.to_model_kwargs()).to(device)
    resume_optimizer = torch.optim.Adam(
        resume_target.parameters(), lr=9e-3, weight_decay=0.0
    )
    resume_lrs_before_load = [
        float(group["lr"]) for group in resume_optimizer.param_groups
    ]
    if resume_optimizer.state_dict()["state"]:
        raise AssertionError("Fresh resume Adam unexpectedly has optimizer state")
    resume_scheduler = torch.optim.lr_scheduler.StepLR(
        resume_optimizer, step_size=3, gamma=0.5
    )
    resume_scaler = train.create_grad_scaler(scaler_enabled)
    resume_generator = torch.Generator().manual_seed(999)
    metadata = train.load_checkpoint(
        full_path,
        resume_target,
        device,
        resume_optimizer,
        resume_scheduler,
        resume_scaler,
        resume_generator,
        expected_model_config=config,
        expected_training_config=train.resume_training_config(args),
        expected_resolved_split=resolved_split,
        amp_enabled=scaler_enabled,
        resume_mode="full",
    )
    _assert_state_equal(resume_target, expected, "full training resume")
    restored_optimizer_state = resume_optimizer.state_dict()
    _assert_nested_equal(
        restored_optimizer_state,
        source_optimizer_state,
        "restored Adam state",
    )
    restored_lrs = [float(group["lr"]) for group in resume_optimizer.param_groups]
    if restored_lrs != source_lrs:
        raise AssertionError(
            f"Optimizer LR did not restore: {restored_lrs} != {source_lrs}"
        )
    _assert_nested_equal(
        resume_scheduler.state_dict(),
        source_scheduler_state,
        "restored scheduler state",
    )
    _assert_nested_equal(
        resume_scaler.state_dict(),
        source_scaler_state,
        "restored disabled scaler state",
    )
    if not torch.equal(resume_generator.get_state(), source_generator_state):
        raise AssertionError("DataLoader generator RNG state did not restore")
    if int(metadata.get("epoch", -1)) != 2 or int(metadata.get("iteration", -1)) != 7:
        raise AssertionError("Full checkpoint epoch/iteration metadata did not round-trip")
    del resume_target

    return {
        "raw_state_dict": raw_path.name,
        "full": full_path.name,
        "state_tensors": len(expected),
        "optimizer_state_entries": len(source_optimizer_state["state"]),
        "adam_moment_tensors": moment_tensor_count,
        "optimizer_lrs_before_restore": resume_lrs_before_load,
        "optimizer_lrs_restored": restored_lrs,
        "scheduler_last_epoch": int(resume_scheduler.last_epoch),
        "scaler_enabled": bool(resume_scaler.is_enabled()),
        "scaler_state_entries": len(resume_scaler.state_dict()),
        "generator_rng_restored": True,
        "format_version": int(payload["checkpoint_format_version"]),
        "artifact_type": str(payload["artifact_type"]),
        "epoch": int(metadata["epoch"]),
        "iteration": int(metadata["iteration"]),
    }


def run_prediction_export_smoke(directory: Path) -> dict[str, Any]:
    """Save and reload one synthetic NIfTI mask through the production exporter."""
    runtime = require_runtime()
    torch = runtime["torch"]
    numpy = runtime["numpy"]
    nibabel = runtime["nibabel"]
    import inference

    shape = (4, 5, 6)
    affine = numpy.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=numpy.float64,
    )
    reference_path = directory / "reference.nii.gz"
    output_path = directory / "prediction.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image(numpy.zeros(shape, dtype=numpy.float32), affine),
        reference_path,
    )
    prediction = torch.zeros(shape, dtype=torch.uint8)
    prediction[1:3, 2:4, 3:5] = 1
    inference.save_prediction(prediction, reference_path, output_path)

    saved = nibabel.load(output_path)
    if tuple(saved.shape) != shape:
        raise AssertionError(f"Exported NIfTI shape {saved.shape} != {shape}")
    if not numpy.array_equal(saved.affine, affine):
        raise AssertionError("Exported NIfTI affine differs from its reference")
    if saved.get_data_dtype() != numpy.dtype(numpy.uint8):
        raise AssertionError(
            f"Exported NIfTI dtype is {saved.get_data_dtype()}, expected uint8"
        )
    if not numpy.array_equal(saved.get_fdata(), prediction.numpy()):
        raise AssertionError("Exported NIfTI voxel values differ from prediction")
    return {
        "shape": list(saved.shape),
        "dtype": str(saved.get_data_dtype()),
        "foreground_voxels": int(prediction.sum().item()),
    }


def run_sliding_window_smoke(
    model: Any, config: Any, *, device: Any, use_amp: bool
) -> dict[str, Any]:
    """Exercise stitching while keeping the complete input/output on CPU."""
    runtime = require_runtime()
    torch = runtime["torch"]
    import inference
    import train

    model.eval()
    predictor = inference.InferencePredictor(model).eval()
    volume_shape = (
        1,
        2,
        config.patch_size[0] + 8,
        config.patch_size[1],
        config.patch_size[2],
    )
    volume = torch.randn(volume_shape, dtype=torch.float32, device="cpu")
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, enabled=use_amp
    ):
        logits = inference.sliding_window_logits(
            volume,
            predictor=predictor,
            roi_size=config.patch_size,
            sw_batch_size=1,
            overlap=0.25,
            compute_device=device,
        )
    expected_shape = (1, 2, *volume_shape[-3:])
    if tuple(logits.shape) != expected_shape:
        raise AssertionError(
            f"Sliding-window output shape {tuple(logits.shape)} != {expected_shape}"
        )
    if logits.device.type != "cpu":
        raise AssertionError(
            f"Sliding-window stitched output must stay on CPU, got {logits.device}"
        )
    _assert_finite(logits, "sliding-window logits")

    # Exercise the separate production validation wrapper with a single window
    # so the server smoke verifies its CPU-stitching contract without doubling
    # the more expensive multi-window pass above.
    validation_ct = volume[:, :1, : config.patch_size[0]]
    validation_pet = volume[:, 1:2, : config.patch_size[0]]
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, enabled=use_amp
    ):
        validation_logits = train.predict_validation_logits(
            model,
            validation_ct,
            validation_pet,
            roi_size=config.patch_size,
            sw_batch_size=1,
            overlap=0.25,
            model_device=device,
        )
    validation_expected = (1, 2, *config.patch_size)
    if tuple(validation_logits.shape) != validation_expected:
        raise AssertionError(
            "Validation sliding-window output shape "
            f"{tuple(validation_logits.shape)} != {validation_expected}"
        )
    if validation_logits.device.type != "cpu":
        raise AssertionError(
            "Validation sliding-window output must stay on CPU, got "
            f"{validation_logits.device}"
        )
    _assert_finite(validation_logits, "validation sliding-window logits")
    return {
        "input_shape": list(volume_shape),
        "output_shape": list(logits.shape),
        "output_device": logits.device.type,
        "validation_output_shape": list(validation_logits.shape),
        "validation_output_device": validation_logits.device.type,
        "overlap": 0.25,
    }


def runtime_information(device: Any, use_amp: bool) -> dict[str, Any]:
    runtime = require_runtime()
    torch = runtime["torch"]
    information: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "monai": str(runtime["monai"].__version__),
        "numpy": str(runtime["numpy"].__version__),
        "scipy": str(runtime["scipy"].__version__),
        "nibabel": str(runtime["nibabel"].__version__),
        "device": str(device),
        "amp": use_amp,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
    }
    if device.type == "cuda":
        information["gpu"] = torch.cuda.get_device_name(device)
    return information


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run strict H2ASeg forward/loss/checkpoint/export/sliding-window "
            "smoke tests"
        )
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:N (auto prefers CUDA)",
    )
    parser.add_argument(
        "--amp",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto enables AMP on CUDA; on requires CUDA",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a machine-readable verification report",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    runtime = require_runtime()
    torch = runtime["torch"]
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    use_amp = resolve_amp(args.amp, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    report: dict[str, Any] = {"runtime": runtime_information(device, use_amp)}
    print(json.dumps(report["runtime"], ensure_ascii=False, sort_keys=True))

    model, config, optimizer, scaler, forward = run_forward_backward(
        device=device, use_amp=use_amp, seed=args.seed
    )
    report["model_config"] = config.to_dict()
    report["forward_backward"] = forward
    print(
        "[PASS] real H2ASeg forward + five-head CE/Dice backward: "
        f"loss={forward['weighted_loss']:.6f}, "
        f"attempts={forward['optimization_attempts']}, "
        f"overflows={forward['overflow_retries']}, "
        f"scale={forward['initial_grad_scale']}->{forward['final_grad_scale']}"
    )

    with tempfile.TemporaryDirectory(prefix="h2aseg-runtime-") as temporary:
        temporary_path = Path(temporary)
        report["checkpoints"] = run_checkpoint_smoke(
            model,
            config,
            optimizer,
            scaler,
            device=device,
            directory=temporary_path,
        )
        report["prediction_export"] = run_prediction_export_smoke(temporary_path)
    print(
        "[PASS] raw/full checkpoint round-trips: "
        f"Adam states={report['checkpoints']['optimizer_state_entries']}, "
        f"restored LR={report['checkpoints']['optimizer_lrs_restored']}"
    )
    print("[PASS] production NIfTI prediction export")

    report["sliding_window"] = run_sliding_window_smoke(
        model, config, device=device, use_amp=use_amp
    )
    print("[PASS] multi-window inference with CPU stitching")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        report["cuda_memory"] = {
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / (1024**2), 2
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved(device) / (1024**2), 2
            ),
        }
        print(
            "[INFO] CUDA peak memory: "
            f"allocated={report['cuda_memory']['peak_allocated_mib']} MiB, "
            f"reserved={report['cuda_memory']['peak_reserved_mib']} MiB"
        )
    report["status"] = "pass"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[INFO] wrote report to {args.json_output}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except RuntimeDependencyError as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    print("[PASS] H2ASeg runtime contract verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
