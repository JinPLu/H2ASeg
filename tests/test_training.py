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
from utils.data import foreground_above_minimum
from utils.model_config import DEFAULT_MODEL_CONFIG, H2ASegModelConfig


def test_cli_accepts_exactly_manifest_or_complete_directories() -> None:
    manifest_args = train.parse_args(["--manifest", "cases.csv"])
    assert manifest_args.manifest == Path("cases.csv")
    assert manifest_args.split_mode == "legacy"
    assert manifest_args.split_seed == 12345

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


def test_transforms_validate_geometry_and_only_crop_training_data() -> None:
    training, validation = train.build_transforms((16, 16, 16))
    training_names = [type(transform).__name__ for transform in training.transforms]
    validation_names = [type(transform).__name__ for transform in validation.transforms]

    assert training_names[:3] == ["LoadImaged", "EnsureChannelFirstd", "ValidateGeometryd"]
    assert training_names.index("ValidateGeometryd") < training_names.index("CropForegroundd")
    assert "CropForegroundd" not in validation_names
    assert not any(name.startswith("Rand") for name in validation_names)

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


def _disabled_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


def test_full_checkpoint_is_weights_only_safe_and_restores_rng(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    scaler = _disabled_scaler()
    generator = torch.Generator().manual_seed(17)
    args = argparse.Namespace(
        manifest=Path("cases.csv"), output_dir=Path("output"), epochs=3, amp=False
    )
    cases = [
        {"case": "A", "ct": "ct.nii.gz", "pet": "pet.nii.gz", "label": "label.nii.gz"}
    ]
    resolved = {"train": cases, "val": cases, "test": cases}

    random.seed(4)
    np.random.seed(5)
    torch.manual_seed(6)
    checkpoint = tmp_path / "full.ckpt"
    train.save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        scaler,
        generator,
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
    assert payload["model_config"] == DEFAULT_MODEL_CONFIG.to_dict()
    assert payload["config"]["manifest"] == "cases.csv"
    assert payload["resolved_cases"] == cases
    assert set(payload["resolved_split"]) == {"train", "val", "test"}

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
    )
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(1),
        torch.rand(1, generator=generator),
    )
    assert restored["epoch"] == 2
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])


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


def test_incomplete_training_wrapper_is_loaded_as_weights_only(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "partial.ckpt"
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 99},
        path,
    )
    assert train.load_checkpoint(path, model, torch.device("cpu")) == {}


def test_legacy_raw_and_wrapped_state_dicts_remain_loadable(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 1)
    expected = {key: value.detach().clone() for key, value in source.state_dict().items()}
    target = torch.nn.Linear(2, 1)

    raw = tmp_path / "raw.pth"
    torch.save(expected, raw)
    assert train.load_checkpoint(raw, target, torch.device("cpu")) == {}
    assert all(torch.equal(target.state_dict()[key], value) for key, value in expected.items())

    wrapped = tmp_path / "wrapped.pth"
    torch.save({"state_dict": {f"module.{key}": value for key, value in expected.items()}}, wrapped)
    assert train.load_checkpoint(wrapped, target, torch.device("cpu")) == {}
    assert all(torch.equal(target.state_dict()[key], value) for key, value in expected.items())
