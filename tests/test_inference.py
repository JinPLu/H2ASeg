"""Focused inference contract tests.

The module is skipped on lightweight development machines without the runtime
stack; the same tests are intended to run unchanged in the server environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
nib = pytest.importorskip("nibabel")

import inference
from utils.data import ValidateGeometryd
from utils.model_config import DEFAULT_MODEL_CONFIG, H2ASegModelConfig


def test_cli_accepts_manifest_or_directory_source() -> None:
    parser = inference.build_parser()
    manifest = parser.parse_args(["--manifest", "cases.csv", "--checkpoint", "model.pth"])
    assert manifest.manifest == Path("cases.csv")
    assert manifest.ct_dir is None
    assert manifest.roi_size is None

    directories = parser.parse_args([
        "--ct-dir", "ct", "--pet-dir", "pet", "--label-dir", "labels",
        "--checkpoint", "model.pth",
    ])
    assert directories.ct_dir == Path("ct")
    assert directories.pet_dir == Path("pet")

    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "model.pth"])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--manifest", "cases.csv", "--ct-dir", "ct", "--checkpoint", "model.pth",
        ])


def test_legacy_checkpoint_uses_validated_default_or_explicit_config() -> None:
    state = {"weight": torch.zeros(1)}
    assert inference.resolve_model_config(state) == DEFAULT_MODEL_CONFIG

    explicit = inference.resolve_model_config(
        state,
        roi_size=(128, 128, 64),
        num_channels=(8, 16, 32, 64, 128),
        window_size=(8, 8, 4),
    )
    assert explicit == DEFAULT_MODEL_CONFIG
    with pytest.raises(ValueError, match="divisible by 16"):
        inference.resolve_model_config(state, roi_size=(120, 128, 64))


def test_full_checkpoint_metadata_is_authoritative() -> None:
    stored = H2ASegModelConfig(patch_size=(128, 96, 64), window_size=(8, 8, 4))
    checkpoint = {"model_state_dict": {"weight": torch.zeros(1)}, "model_config": stored.to_dict()}
    assert inference.resolve_model_config(checkpoint) == stored
    assert inference.resolve_model_config(checkpoint, roi_size=stored.patch_size) == stored

    with pytest.raises(ValueError, match=r"conflicts.*patch_size"):
        inference.resolve_model_config(checkpoint, roi_size=(128, 128, 64))
    with pytest.raises(ValueError, match="invalid model_config"):
        inference.resolve_model_config({**checkpoint, "model_config": {"patch_size": [1]}})


def test_checkpoint_loader_accepts_raw_full_and_repeated_prefixes(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 1)
    expected = {key: value.detach().clone() for key, value in source.state_dict().items()}
    payloads = {
        "raw.pth": expected,
        "full.ckpt": {
            "model_state_dict": expected,
            "model_config": DEFAULT_MODEL_CONFIG.to_dict(),
        },
        "prefixed.pth": {f"module.model.{key}": value for key, value in expected.items()},
    }
    for filename, payload in payloads.items():
        path = tmp_path / filename
        torch.save(payload, path)
        target = torch.nn.Linear(2, 1)
        inference.load_checkpoint(target, path, torch.device("cpu"))
        assert all(
            torch.equal(target.state_dict()[key], value)
            for key, value in expected.items()
        )


def _save_nifti(path: Path, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), affine), path)


def test_inference_transform_checks_geometry_after_load_and_channel_first(tmp_path: Path) -> None:
    transforms = inference.build_inference_transform(("ct", "pet", "label"))
    assert [type(item).__name__ for item in transforms.transforms] == [
        "LoadImaged", "EnsureChannelFirstd", "ValidateGeometryd",
    ]
    assert isinstance(transforms.transforms[-1], ValidateGeometryd)

    identity = np.eye(4)
    shifted = identity.copy()
    shifted[0, 3] = 2.0
    ct, pet, label = (tmp_path / name for name in ("ct.nii.gz", "pet.nii.gz", "label.nii.gz"))
    _save_nifti(ct, identity)
    _save_nifti(pet, identity)
    _save_nifti(label, shifted)
    with pytest.raises(RuntimeError) as error:
        transforms({"case": "MISALIGNED", "ct": str(ct), "pet": str(pet), "label": str(label)})
    assert "affine differs" in str(error.value.__cause__)


def test_inference_transform_rejects_4d_multichannel_nifti(tmp_path: Path) -> None:
    affine = np.eye(4)
    ct = tmp_path / "ct.nii.gz"
    pet = tmp_path / "pet.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((8, 8, 8, 2), dtype=np.float32), affine),
        ct,
    )
    _save_nifti(pet, affine)
    transforms = inference.build_inference_transform(("ct", "pet"))
    with pytest.raises(RuntimeError) as error:
        transforms({"case": "MULTICHANNEL", "ct": str(ct), "pet": str(pet)})
    assert "must have exactly one channel" in str(error.value.__cause__)


def test_prediction_export_preserves_affine_and_header_dtype(tmp_path: Path) -> None:
    affine = np.array([
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 3.0, 0.0, 20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    reference = tmp_path / "reference.nii.gz"
    output = tmp_path / "prediction.nii.gz"
    _save_nifti(reference, affine)

    inference.save_prediction(torch.ones((8, 8, 8)), reference, output)
    saved = nib.load(output)
    assert np.array_equal(saved.affine, affine)
    assert saved.get_data_dtype() == np.dtype(np.uint8)


def test_hausdorff95_uses_physical_spacing_and_handles_empty_masks() -> None:
    label = torch.zeros((1, 1, 4, 4, 4), dtype=torch.bool)
    prediction = torch.zeros_like(label)
    label[0, 0, 1, 1, 1] = True
    prediction[0, 0, 2, 1, 1] = True
    assert inference.hausdorff95(label, prediction, (2.0, 1.0, 1.0)) == pytest.approx(2.0)
    assert np.isnan(inference.hausdorff95(torch.zeros_like(label), prediction, (1, 1, 1)))
