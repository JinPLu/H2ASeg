"""Strict aggregate contracts for a fully installed H2ASeg runtime.

Unlike the lightweight unit-test modules, this file must not silently skip when
the ML stack is absent.  It is the pytest companion to
``scripts/verify_runtime.py`` and is intended for the configured server.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pytest

from scripts.verify_runtime import (
    RuntimeDependencyError,
    require_runtime,
    resolve_amp,
    resolve_device,
    run_checkpoint_smoke,
    run_forward_backward,
    run_prediction_export_smoke,
    run_sliding_window_smoke,
)


try:
    RUNTIME = require_runtime()
except RuntimeDependencyError as exc:
    raise RuntimeError(
        "Strict H2ASeg runtime contracts require the complete ML stack; "
        "this module is failing instead of being skipped.\n"
        f"{exc}"
    ) from exc


@pytest.fixture(scope="module")
def runtime_bundle() -> dict[str, Any]:
    device = resolve_device(os.environ.get("H2ASEG_TEST_DEVICE", "cpu"))
    use_amp = resolve_amp(os.environ.get("H2ASEG_TEST_AMP", "auto"), device)
    model, config, optimizer, scaler, result = run_forward_backward(
        device=device,
        use_amp=use_amp,
        seed=12345,
    )
    return {
        "device": device,
        "use_amp": use_amp,
        "model": model,
        "config": config,
        "optimizer": optimizer,
        "scaler": scaler,
        "forward_backward": result,
    }


def test_real_h2aseg_small_config_forward(runtime_bundle: dict[str, Any]) -> None:
    result = runtime_bundle["forward_backward"]
    expected_shape = (1, 2, 32, 32, 32)
    assert result["output_shapes"] == [expected_shape] * 5
    assert result["weighted_loss"] > 0


def test_real_five_head_ce_dice_backward(runtime_bundle: dict[str, Any]) -> None:
    result = runtime_bundle["forward_backward"]
    assert len(result["head_losses"]) == 5
    assert all(value > 0 for value in result["head_losses"])
    assert result["parameters_with_grad"] > 0
    assert result["heads_with_grad"] == [1, 2, 3, 4, 5]
    assert result["changed_parameters"] > 0
    assert result["optimizer_step_verified"] is True
    assert result["amp"] is runtime_bundle["use_amp"]
    assert result["optimization_attempts"] == result["overflow_retries"] + 1
    assert 0 <= result["overflow_retries"] <= 7
    assert len(result["overflow_events"]) == result["overflow_retries"]
    for expected_attempt, event in enumerate(result["overflow_events"], 1):
        assert event["attempt"] == expected_attempt
        assert event["bad_gradient_tensors"] > 0
        assert event["bad_gradient_elements"] > 0
        assert event["scale_after"] < event["scale_before"]
        assert event["optimizer_step_skipped"] is True
        assert isinstance(event["forced"], bool)
        if expected_attempt > 1:
            previous = result["overflow_events"][expected_attempt - 2]
            assert event["scale_before"] == pytest.approx(previous["scale_after"])
    if runtime_bundle["use_amp"]:
        assert result["final_grad_scale"] <= result["initial_grad_scale"]
    else:
        assert result["optimization_attempts"] == 1
        assert result["overflow_retries"] == 0
        assert result["initial_grad_scale"] == pytest.approx(1.0)
        assert result["final_grad_scale"] == pytest.approx(1.0)


def test_forced_cuda_amp_overflow_warms_down_then_updates() -> None:
    torch = RUNTIME["torch"]
    if not torch.cuda.is_available():
        pytest.fail(
            "Strict runtime contracts require CUDA for the forced AMP overflow test"
        )

    model, config, optimizer, scaler, result = run_forward_backward(
        device=torch.device("cuda"),
        use_amp=True,
        seed=12345,
        force_overflow_attempts=1,
    )
    assert result["overflow_retries"] >= 1
    assert result["optimization_attempts"] == result["overflow_retries"] + 1
    assert result["optimization_attempts"] <= 8
    first = result["overflow_events"][0]
    assert first["attempt"] == 1
    assert first["forced"] is True
    assert first["optimizer_step_skipped"] is True
    assert first["bad_gradient_tensors"] >= 1
    assert first["bad_gradient_elements"] >= 1
    assert first["scale_after"] < first["scale_before"]
    assert result["final_grad_scale"] < result["initial_grad_scale"]
    assert all(math.isfinite(value) for value in result["head_losses"])
    assert math.isfinite(result["weighted_loss"])
    assert result["changed_parameters"] > 0
    assert result["optimizer_step_verified"] is True

    # Keep references alive through every assertion, then release GPU storage.
    del model, config, optimizer, scaler
    torch.cuda.empty_cache()


def test_forced_overflow_control_is_bounded_and_cuda_amp_only() -> None:
    torch = RUNTIME["torch"]
    with pytest.raises(ValueError, match=r"integer in \[0, 7\]"):
        run_forward_backward(
            device=torch.device("cpu"),
            use_amp=False,
            force_overflow_attempts=8,
        )
    with pytest.raises(ValueError, match="requires AMP on CUDA"):
        run_forward_backward(
            device=torch.device("cpu"),
            use_amp=False,
            force_overflow_attempts=1,
        )


def test_real_raw_state_dict_and_full_checkpoint_round_trip(
    runtime_bundle: dict[str, Any], tmp_path: Path
) -> None:
    result = run_checkpoint_smoke(
        runtime_bundle["model"],
        runtime_bundle["config"],
        runtime_bundle["optimizer"],
        runtime_bundle["scaler"],
        device=runtime_bundle["device"],
        directory=tmp_path,
    )
    assert result["state_tensors"] > 0
    assert result["optimizer_state_entries"] > 0
    assert result["adam_moment_tensors"] >= 2
    assert result["optimizer_lrs_before_restore"] == pytest.approx([9e-3])
    assert result["optimizer_lrs_restored"] == pytest.approx([1e-4])
    assert result["scheduler_last_epoch"] == 0
    assert result["scaler_enabled"] is runtime_bundle["use_amp"]
    if runtime_bundle["use_amp"]:
        assert result["scaler_state_entries"] > 0
    else:
        assert result["scaler_state_entries"] == 0
    assert result["generator_rng_restored"] is True
    assert result["format_version"] >= 1
    assert result["artifact_type"] == "h2aseg_training_checkpoint"
    assert result["epoch"] == 2
    assert result["iteration"] == 7


def test_production_nifti_prediction_export(tmp_path: Path) -> None:
    result = run_prediction_export_smoke(tmp_path)
    assert result == {
        "shape": [4, 5, 6],
        "dtype": "uint8",
        "foreground_voxels": 8,
    }


def test_real_multi_window_inference_keeps_stitching_on_cpu(
    runtime_bundle: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inference
    import train

    calls = {"inference": 0, "validation": 0}
    inference_wrapper = inference.sliding_window_logits
    validation_wrapper = train.predict_validation_logits

    def traced_inference(*args: Any, **kwargs: Any) -> Any:
        calls["inference"] += 1
        return inference_wrapper(*args, **kwargs)

    def traced_validation(*args: Any, **kwargs: Any) -> Any:
        calls["validation"] += 1
        return validation_wrapper(*args, **kwargs)

    monkeypatch.setattr(inference, "sliding_window_logits", traced_inference)
    monkeypatch.setattr(train, "predict_validation_logits", traced_validation)
    result = run_sliding_window_smoke(
        runtime_bundle["model"],
        runtime_bundle["config"],
        device=runtime_bundle["device"],
        use_amp=runtime_bundle["use_amp"],
    )
    assert result["input_shape"] == [1, 2, 40, 32, 32]
    assert result["output_shape"] == [1, 2, 40, 32, 32]
    assert result["output_device"] == "cpu"
    assert result["validation_output_shape"] == [1, 2, 32, 32, 32]
    assert result["validation_output_device"] == "cpu"
    assert calls == {"inference": 1, "validation": 1}
