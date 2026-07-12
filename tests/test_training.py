"""Focused training-contract tests; requires the documented runtime stack."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("tensorboard")

import train
from utils.data import foreground_above_minimum, read_manifest, split_cases
from utils.model_config import DEFAULT_MODEL_CONFIG, H2ASegModelConfig


def test_cli_accepts_exactly_manifest_or_complete_directories() -> None:
    manifest_args = train.parse_args(["--manifest", "cases.csv"])
    assert manifest_args.manifest == Path("cases.csv")
    assert manifest_args.split_mode == "legacy"
    assert manifest_args.split_seed == 12345
    assert manifest_args.resume_mode == "auto"

    directory_args = train.parse_args(
        ["--ct-dir", "ct", "--pet-dir", "pet", "--label-dir", "labels"]
    )
    assert directory_args.label_dir == Path("labels")

    with pytest.raises(SystemExit):
        train.parse_args(["--ct-dir", "ct", "--pet-dir", "pet"])
    with pytest.raises(SystemExit):
        train.parse_args(
            [
                "--manifest",
                "cases.csv",
                "--ct-dir",
                "ct",
                "--pet-dir",
                "pet",
                "--label-dir",
                "labels",
            ]
        )


def test_label_binarization_precedes_integer_conversion() -> None:
    labels = torch.tensor([[[[[0.0, 0.5, -0.5, 2.0]]]]])
    result = train.binarize_labels(labels, torch.device("cpu"))
    assert result.dtype == torch.long
    assert result.tolist() == [[[[[0, 1, 1, 1]]]]]


def test_validation_cadence_always_includes_final_epoch() -> None:
    assert not train.should_validate_epoch(0, total_epochs=3, interval=10)
    assert not train.should_validate_epoch(1, total_epochs=3, interval=10)
    assert train.should_validate_epoch(2, total_epochs=3, interval=10)
    assert train.should_validate_epoch(9, total_epochs=20, interval=10)


def test_validation_summary_excludes_negative_gt_from_overlap_selection() -> None:
    summary = train.summarize_validation_cases(
        [
            (True, (0.1, 0.2, 0.3, 0.4)),
            # Correct lesion-free prediction: standalone Dice/IoU are one, but
            # these values must not raise the positive-case model-selection score.
            (False, (0.0, 0.0, 1.0, 1.0)),
            (True, (0.3, 0.4, 0.5, 0.6)),
            (False, (0.2, 0.0, 0.0, 0.0)),
        ]
    )
    np.testing.assert_allclose(summary.positive_mean, (0.2, 0.3, 0.4, 0.5))
    assert summary.positive_cases == 2
    assert summary.negative_cases == 2
    assert summary.negative_mean_fpr == pytest.approx(0.1)


def test_validation_summary_rejects_all_negative_gt() -> None:
    with pytest.raises(ValueError, match="no positive-GT cases"):
        train.summarize_validation_cases(
            [(False, (0.0, 0.0, 1.0, 1.0))]
        )


def test_resumed_best_dice_requires_matching_validation_policy() -> None:
    assert train.resolve_resumed_best_dice(
        {
            "best_dice": 0.7,
            "validation_metric_policy": train.VALIDATION_METRIC_POLICY,
        }
    ) == pytest.approx(0.7)
    with pytest.warns(RuntimeWarning, match="incompatible validation metric policy"):
        assert train.resolve_resumed_best_dice({"best_dice": 0.9}) == 0.0


def test_non_finite_loss_fails_before_backward() -> None:
    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        train.ensure_finite_losses(
            torch.tensor(float("nan")), [torch.tensor(1.0), torch.tensor(float("nan"))]
        )


def test_validation_sliding_window_keeps_full_tensors_on_cpu(monkeypatch) -> None:
    observed = {}

    def fake_sliding_window(inputs, *, predictor, sw_device, device, **kwargs):
        observed.update(
            input_device=inputs.device,
            sw_device=sw_device,
            output_device=device,
        )
        return predictor(inputs.to(sw_device)).to(device)

    class TwoChannelModel(torch.nn.Module):
        def forward(self, ct, pet):
            return [torch.cat((ct, pet), dim=1)]

    monkeypatch.setattr(train, "sliding_window_inference", fake_sliding_window)
    logits = train.predict_validation_logits(
        TwoChannelModel(),
        torch.ones(1, 1, 4, 4, 4),
        torch.zeros(1, 1, 4, 4, 4),
        roi_size=(4, 4, 4),
        sw_batch_size=1,
        overlap=0.5,
        model_device=torch.device("cpu"),
    )
    assert observed == {
        "input_device": torch.device("cpu"),
        "sw_device": torch.device("cpu"),
        "output_device": torch.device("cpu"),
    }
    assert logits.device.type == "cpu"


def test_transforms_validate_geometry_and_only_crop_training_data() -> None:
    training, validation = train.build_transforms((16, 16, 16))
    training_names = [type(transform).__name__ for transform in training.transforms]
    validation_names = [type(transform).__name__ for transform in validation.transforms]

    assert training_names[:3] == ["LoadImaged", "EnsureChannelFirstd", "ValidateGeometryd"]
    assert training_names.index("ValidateGeometryd") < training_names.index("CropForegroundd")
    assert "CropForegroundd" not in validation_names
    assert not any(name.startswith("Rand") for name in validation_names)
    assert all(
        train_item is not validation_item
        for train_item, validation_item in zip(
            training.transforms[:3], validation.transforms[:3]
        )
    )

    training_validator = training.transforms[2]
    validation_validator = validation.transforms[2]
    assert training_validator.validate_pet_crop is True
    assert validation_validator.validate_pet_crop is False

    crop = next(
        transform for transform in training.transforms if type(transform).__name__ == "CropForegroundd"
    )
    assert crop.cropper.select_fn is foreground_above_minimum


def test_resolved_split_csv_contains_all_partitions(tmp_path: Path) -> None:
    def case(name: str) -> dict[str, str]:
        return {
            "case": name,
            "ct": f"/{name}_ct.nii.gz",
            "pet": f"/{name}_pet.nii.gz",
            "label": f"/{name}_label.nii.gz",
        }

    snapshot = train.resolved_split_snapshot([case("A")], [case("B")], [case("C")])
    destination = tmp_path / "run" / "resolved_split.csv"
    train.write_resolved_split(destination, snapshot)

    with destination.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["case"], row["split"]) for row in rows] == [
        ("A", "train"),
        ("B", "val"),
        ("C", "test"),
    ]

    train.validate_resume_split(snapshot, snapshot)
    changed = train.resolved_split_snapshot([case("C")], [case("B")], [case("A")])
    with pytest.raises(ValueError, match="resolved data split differs"):
        train.validate_resume_split(snapshot, changed)


@pytest.mark.parametrize("with_group", [False, True])
def test_resolved_split_csv_round_trips_for_exact_resume(
    tmp_path: Path, with_group: bool
) -> None:
    def case(name: str, split_name: str) -> dict[str, str]:
        record = {"case": name}
        for modality in ("ct", "pet", "label"):
            path = tmp_path / f"{name}_{modality}.nii.gz"
            path.touch()
            record[modality] = str(path)
        if with_group:
            record["group"] = f"patient-{name}"
            record["split"] = split_name
        return record

    original = train.resolved_split_snapshot(
        [case("A", "train")],
        [case("B", "val")],
        [case("C", "test")],
    )
    destination = tmp_path / "resolved.csv"
    train.write_resolved_split(destination, original)

    with destination.open(newline="", encoding="utf-8") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    assert ("group" in fieldnames) is with_group

    reloaded = read_manifest(destination, require_labels=True)
    reloaded_split = train.resolved_split_snapshot(
        *split_cases(reloaded, mode="legacy", seed=12345)
    )
    train.validate_resume_split(original, reloaded_split)


def _disabled_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


def test_atomic_torch_save_preserves_existing_target_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "artifact.pth"
    destination.write_bytes(b"known-good")

    def interrupted_save(payload, handle):
        handle.write(b"partial")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(train.torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train.atomic_torch_save({"value": 1}, destination)
    assert destination.read_bytes() == b"known-good"
    assert not list(tmp_path.glob(".artifact.pth.*.tmp"))


def test_full_checkpoint_is_weights_only_safe_and_restores_rng(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    scaler = _disabled_scaler()
    generator = torch.Generator().manual_seed(17)
    train_transform, _ = train.build_transforms((16, 16, 16))
    args = argparse.Namespace(
        manifest=Path("cases.csv"),
        output_dir=Path("output"),
        epochs=3,
        amp=False,
        batch_size=2,
        num_workers=0,
        seed=12345,
        lr=1e-4,
        split_mode="legacy",
        split_seed=12345,
        val_interval=10,
        val_sw_batch_size=4,
        val_overlap=0.5,
    )
    cases = [
        {"case": "A", "ct": "ct.nii.gz", "pet": "pet.nii.gz", "label": "label.nii.gz"}
    ]
    resolved = {"train": cases, "val": cases, "test": cases}

    random.seed(4)
    np.random.seed(5)
    torch.manual_seed(6)
    with pytest.raises(ValueError, match="without all resume configuration fields"):
        train.save_checkpoint(
            tmp_path / "invalid-save.ckpt",
            model,
            optimizer,
            scheduler,
            scaler,
            generator,
            epoch=2,
            iteration=9,
            best_dice=0.7,
            best_train_dice=0.8,
            args=argparse.Namespace(lr=1e-4),
            model_config=DEFAULT_MODEL_CONFIG,
            resolved_cases=cases,
            resolved_split=resolved,
        )
    checkpoint = tmp_path / "full.ckpt"
    train.save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        scaler,
        generator,
        train_transform,
        epoch=2,
        iteration=9,
        best_dice=0.7,
        best_train_dice=0.8,
        args=args,
        model_config=DEFAULT_MODEL_CONFIG,
        resolved_cases=cases,
        resolved_split=resolved,
    )
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(1),
        torch.rand(1, generator=generator),
    )

    # The inference path deliberately uses the restricted loader.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["checkpoint_format_version"] == train.CHECKPOINT_FORMAT_VERSION
    assert payload["artifact_type"] == train.CHECKPOINT_ARTIFACT_TYPE
    assert payload["amp_enabled"] is False
    assert payload["validation_metric_policy"] == train.VALIDATION_METRIC_POLICY
    assert payload["rng_state"]["transform"]
    assert payload["model_config"] == DEFAULT_MODEL_CONFIG.to_dict()
    assert payload["config"]["manifest"] == "cases.csv"
    assert payload["resolved_cases"] == cases
    assert set(payload["resolved_split"]) == {"train", "val", "test"}
    assert train.FULL_RESUME_KEYS.issubset(payload)

    missing_artifact = dict(payload)
    missing_artifact.pop("artifact_type")
    missing_artifact_path = tmp_path / "missing-artifact.ckpt"
    torch.save(missing_artifact, missing_artifact_path)
    with pytest.raises(ValueError, match="Versioned full checkpoint.*artifact_type"):
        train.load_checkpoint(
            missing_artifact_path, torch.nn.Linear(2, 1), torch.device("cpu")
        )

    invalid_required_values = (
        ("checkpoint_format_version", None, "checkpoint_format_version=None"),
        ("artifact_type", None, "artifact_type=None"),
        ("amp_enabled", None, "amp_enabled metadata must be a boolean"),
        (
            "validation_metric_policy",
            None,
            "validation_metric_policy must be a non-empty string",
        ),
    )
    for field, value, message in invalid_required_values:
        invalid = dict(payload)
        invalid[field] = value
        invalid_path = tmp_path / f"invalid-{field}.ckpt"
        torch.save(invalid, invalid_path)
        with pytest.raises(ValueError, match=message):
            train.load_checkpoint(
                invalid_path, torch.nn.Linear(2, 1), torch.device("cpu")
            )

    incomplete_config = dict(payload)
    incomplete_config["config"] = dict(payload["config"])
    incomplete_config["config"].pop("seed")
    incomplete_config_path = tmp_path / "incomplete-config.ckpt"
    torch.save(incomplete_config, incomplete_config_path)
    with pytest.raises(ValueError, match="config is missing resume fields: seed"):
        train.load_checkpoint(
            incomplete_config_path, torch.nn.Linear(2, 1), torch.device("cpu")
        )

    invalid_rng = dict(payload)
    invalid_rng["rng_state"] = None
    invalid_rng_path = tmp_path / "invalid-rng.ckpt"
    torch.save(invalid_rng, invalid_rng_path)
    with pytest.raises(ValueError, match="rng_state must be a mapping"):
        train.load_checkpoint(
            invalid_rng_path, torch.nn.Linear(2, 1), torch.device("cpu")
        )

    random.seed(40)
    np.random.seed(50)
    torch.manual_seed(60)
    generator.manual_seed(70)
    restored = train.load_checkpoint(
        checkpoint,
        model,
        torch.device("cpu"),
        optimizer,
        scheduler,
        scaler,
        generator,
        expected_model_config=DEFAULT_MODEL_CONFIG,
        train_transform=train_transform,
        expected_training_config=train.resume_training_config(args),
        expected_resolved_split=resolved,
        amp_enabled=False,
    )
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(1),
        torch.rand(1, generator=generator),
    )
    assert restored["epoch"] == 2
    assert restored["validation_metric_policy"] == train.VALIDATION_METRIC_POLICY
    assert "model_state_dict" not in restored
    assert "optimizer_state_dict" not in restored
    assert "rng_state" not in restored
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])

    with pytest.raises(ValueError, match="cannot change AMP mode"):
        train.load_checkpoint(
            checkpoint,
            model,
            torch.device("cpu"),
            amp_enabled=True,
        )
    with pytest.raises(ValueError, match="configuration conflicts"):
        train.load_checkpoint(
            checkpoint,
            model,
            torch.device("cpu"),
            expected_training_config={"lr": 2e-4},
        )
    untouched = torch.nn.Linear(2, 1)
    before = {
        key: value.detach().clone() for key, value in untouched.state_dict().items()
    }
    different_split = {"train": [], "val": cases, "test": cases}
    with pytest.raises(ValueError, match="resolved data split differs"):
        train.load_checkpoint(
            checkpoint,
            untouched,
            torch.device("cpu"),
            expected_resolved_split=different_split,
        )
    assert all(
        torch.equal(untouched.state_dict()[key], value)
        for key, value in before.items()
    )

    # Checkpoints produced by the previous schema had no explicit version,
    # artifact type, or AMP flag. Their scaler state still makes AMP inferable.
    legacy_payload = dict(payload)
    legacy_payload.pop("checkpoint_format_version")
    legacy_payload.pop("artifact_type")
    legacy_payload.pop("amp_enabled")
    legacy_payload.pop("validation_metric_policy")
    legacy_checkpoint = tmp_path / "legacy-full.ckpt"
    torch.save(legacy_payload, legacy_checkpoint)
    legacy_model = torch.nn.Linear(2, 1)
    legacy_optimizer = torch.optim.Adam(legacy_model.parameters(), lr=1e-4)
    legacy_scheduler = torch.optim.lr_scheduler.StepLR(
        legacy_optimizer, step_size=1, gamma=0.99
    )
    with pytest.warns(RuntimeWarning, match="versionless previous full"):
        legacy_restored = train.load_checkpoint(
            legacy_checkpoint,
            legacy_model,
            torch.device("cpu"),
            legacy_optimizer,
            legacy_scheduler,
            _disabled_scaler(),
            torch.Generator(),
            amp_enabled=False,
        )
    assert legacy_restored["epoch"] == 2


def test_transform_rng_state_round_trip() -> None:
    class Randomizable:
        def __init__(self, seed):
            self.R = np.random.RandomState(seed)

    class Pipeline(Randomizable):
        def __init__(self):
            super().__init__(1)
            self.transforms = [Randomizable(2), Randomizable(3)]

    pipeline = Pipeline()
    saved = train.capture_transform_rng_state(pipeline)
    expected = [
        rng.random_sample()
        for _, _, rng in train._iter_transform_random_states(pipeline)
    ]
    for _, _, rng in train._iter_transform_random_states(pipeline):
        rng.random_sample(5)
    train.restore_transform_rng_state(pipeline, saved)
    actual = [
        rng.random_sample()
        for _, _, rng in train._iter_transform_random_states(pipeline)
    ]
    assert actual == expected


def test_checkpoint_model_configuration_mismatch_fails_before_training(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "mismatch.ckpt"
    different = H2ASegModelConfig(attention_dropout=0.25)
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": different.to_dict()},
        path,
    )
    with pytest.raises(ValueError, match="model_config does not match"):
        train.load_checkpoint(
            path,
            model,
            torch.device("cpu"),
            expected_model_config=DEFAULT_MODEL_CONFIG,
        )


@pytest.mark.parametrize("wrapper_key", ["model_state_dict", "state_dict", "model"])
def test_incomplete_training_wrapper_requires_explicit_weights_only(
    tmp_path: Path, wrapper_key: str
) -> None:
    model = torch.nn.Linear(2, 1)
    path = tmp_path / f"partial-{wrapper_key}.ckpt"
    torch.save(
        {wrapper_key: model.state_dict(), "epoch": 99},
        path,
    )
    with pytest.raises(ValueError, match="partial training metadata"):
        train.load_checkpoint(path, model, torch.device("cpu"))
    assert (
        train.load_checkpoint(
            path,
            model,
            torch.device("cpu"),
            resume_mode="weights-only",
        )
        == {}
    )


def test_legacy_raw_and_wrapped_state_dicts_remain_loadable(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 1)
    expected = {key: value.detach().clone() for key, value in source.state_dict().items()}
    repeated = {
        f"module.model._model.{key}": value for key, value in expected.items()
    }
    payloads = {
        "raw.pth": expected,
        "state-wrapper.pth": {"state_dict": repeated},
        "model-wrapper.pth": {"model": repeated},
        "model-state-wrapper.pth": {"model_state_dict": repeated},
    }
    for filename, payload in payloads.items():
        path = tmp_path / filename
        torch.save(payload, path)
        target = torch.nn.Linear(2, 1)
        assert train.load_checkpoint(path, target, torch.device("cpu")) == {}
        assert all(
            torch.equal(target.state_dict()[key], value)
            for key, value in expected.items()
        )

    raw = tmp_path / "raw.pth"
    target = torch.nn.Linear(2, 1)
    collision = {
        "weight": expected["weight"],
        "module.weight": expected["weight"],
        "bias": expected["bias"],
    }
    collision_path = tmp_path / "collision.pth"
    torch.save(collision, collision_path)
    with pytest.raises(ValueError, match="key collision"):
        train.load_checkpoint(collision_path, target, torch.device("cpu"))

    with pytest.raises(ValueError, match="cannot be used for full resume"):
        train.load_checkpoint(
            raw, target, torch.device("cpu"), resume_mode="full"
        )
