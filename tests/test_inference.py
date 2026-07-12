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


def test_checkpoint_payload_is_always_loaded_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_load(path: Path, **kwargs: object) -> dict[str, torch.Tensor]:
        calls.append(kwargs)
        return {"weight": torch.zeros(1)}

    monkeypatch.setattr(torch, "load", fake_load)
    inference._load_checkpoint_file(Path("full.ckpt"))
    assert calls == [{"map_location": torch.device("cpu"), "weights_only": True}]


def test_checkpoint_loader_rejects_prefix_normalization_collisions() -> None:
    target = torch.nn.Linear(2, 1)
    checkpoint = {
        "weight": torch.ones_like(target.weight),
        "module.weight": torch.zeros_like(target.weight),
        "bias": torch.zeros_like(target.bias),
    }
    with pytest.raises(ValueError, match=r"keys collide.*weight.*module\.weight"):
        inference.load_checkpoint(
            target,
            Path("unused.ckpt"),
            torch.device("cpu"),
            checkpoint=checkpoint,
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


def test_prediction_export_preserves_singleton_axis_and_rejects_shape_mismatch(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.nii.gz"
    output = tmp_path / "prediction.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((8, 8, 1), dtype=np.float32), np.eye(4)),
        reference,
    )

    inference.save_prediction(torch.ones((8, 8, 1)), reference, output)
    assert nib.load(output).shape == (8, 8, 1)

    with pytest.raises(ValueError, match="exactly 3D"):
        inference.save_prediction(torch.ones((1, 1, 8, 8, 1)), reference, output)
    with pytest.raises(ValueError, match="shape mismatch"):
        inference.save_prediction(torch.ones((8, 7, 1)), reference, output)


@pytest.mark.parametrize("case", ["", ".", "..", "../escape", "nested/case", r"nested\case"])
def test_prediction_output_path_rejects_unsafe_case_identifiers(
    tmp_path: Path, case: str
) -> None:
    with pytest.raises(ValueError, match="Case identifier|Unsafe case identifier"):
        inference.prediction_output_path(tmp_path / "predictions", case)


def test_prediction_output_path_is_contained(tmp_path: Path) -> None:
    root = tmp_path / "predictions"
    output = inference.prediction_output_path(
        root,
        "case-001",
        protected_paths=[tmp_path / "ct" / "case-001.nii.gz"],
    )
    assert output == root.resolve() / "case-001.nii.gz"


@pytest.mark.parametrize("source_kind", ["ct", "pet", "label"])
def test_prediction_output_path_refuses_to_overwrite_case_sources(
    tmp_path: Path, source_kind: str
) -> None:
    source_dir = tmp_path / source_kind
    protected = [
        tmp_path / "other" / "unrelated.nii.gz",
        source_dir / "case-001.nii.gz",
    ]
    with pytest.raises(ValueError, match="Refusing to overwrite case source image"):
        inference.prediction_output_path(
            source_dir,
            "case-001",
            protected_paths=protected,
        )


def test_prediction_output_preflight_rejects_cross_case_source_collision(
    tmp_path: Path,
) -> None:
    prediction_dir = tmp_path / "predictions"
    records = [
        {
            "case": "case-a",
            "ct": str(tmp_path / "ct" / "unrelated-a.nii.gz"),
            "pet": str(tmp_path / "pet" / "unrelated-a.nii.gz"),
        },
        {
            "case": "case-b",
            "ct": str(prediction_dir / "case-a.nii.gz"),
            "pet": str(tmp_path / "pet" / "unrelated-b.nii.gz"),
            "label": str(tmp_path / "labels" / "unrelated-b.nii.gz"),
        },
    ]
    with pytest.raises(ValueError, match="Refusing to overwrite case source image"):
        inference.prepare_prediction_output_paths(records, prediction_dir)


def test_prediction_output_preflight_keeps_normal_destinations(tmp_path: Path) -> None:
    records = [
        {
            "case": "case-a",
            "ct": str(tmp_path / "ct" / "case-a.nii.gz"),
            "pet": str(tmp_path / "pet" / "case-a.nii.gz"),
        },
        {
            "case": "case-b",
            "ct": str(tmp_path / "ct" / "case-b.nii.gz"),
            "pet": str(tmp_path / "pet" / "case-b.nii.gz"),
        },
    ]
    prediction_dir = tmp_path / "predictions"
    assert inference.prepare_prediction_output_paths(records, prediction_dir) == {
        "case-a": prediction_dir.resolve() / "case-a.nii.gz",
        "case-b": prediction_dir.resolve() / "case-b.nii.gz",
    }


def test_prediction_export_refuses_to_overwrite_reference(tmp_path: Path) -> None:
    reference = tmp_path / "case-001.nii.gz"
    _save_nifti(reference, np.eye(4))
    with pytest.raises(ValueError, match="Refusing to overwrite source image"):
        inference.save_prediction(torch.ones((8, 8, 8)), reference, reference)
    assert not np.any(nib.load(reference).get_fdata())


def test_sliding_window_keeps_full_volume_and_stitched_output_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_sliding_window(inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
        observed["input_device"] = inputs.device
        observed.update(kwargs)
        return torch.zeros((1, 2, *inputs.shape[2:]), device=kwargs["device"])

    monkeypatch.setattr(inference, "sliding_window_inference", fake_sliding_window)
    compute_device = torch.device("cuda:7")
    inputs = torch.zeros((1, 2, 8, 8, 8))
    logits = inference.sliding_window_logits(
        inputs,
        predictor=torch.nn.Identity(),
        roi_size=(8, 8, 8),
        sw_batch_size=2,
        overlap=0.5,
        compute_device=compute_device,
    )

    assert observed["input_device"] == torch.device("cpu")
    assert observed["sw_device"] == compute_device
    assert observed["device"] == torch.device("cpu")
    assert logits.device == torch.device("cpu")


@pytest.mark.parametrize(
    ("label_values", "prediction_values", "expected_precision", "expected_recall"),
    [
        ([0, 0, 0, 0], [0, 0, 0, 0], None, None),
        ([1, 0, 0, 0], [0, 0, 0, 0], None, 0.0),
        ([0, 0, 0, 0], [1, 0, 0, 0], 0.0, None),
        ([1, 1, 0, 0], [1, 0, 1, 0], 0.5, 0.5),
    ],
)
def test_binary_metrics_precision_recall_empty_semantics(
    label_values: list[int],
    prediction_values: list[int],
    expected_precision: float | None,
    expected_recall: float | None,
) -> None:
    label = torch.tensor(label_values, dtype=torch.bool).reshape(1, 1, 2, 2)
    prediction = torch.tensor(prediction_values, dtype=torch.bool).reshape(1, 1, 2, 2)
    result = inference.binary_metrics(label, prediction)
    canonical = inference.metrics_tensor(label, prediction)

    assert set(result) == {
        "false_positive_rate",
        "false_negative_rate",
        "iou",
        "dice",
        "precision",
        "recall",
    }
    assert (
        result["false_positive_rate"],
        result["false_negative_rate"],
        result["iou"],
        result["dice"],
    ) == pytest.approx(canonical)
    if expected_precision is None:
        assert np.isnan(result["precision"])
    else:
        assert result["precision"] == pytest.approx(expected_precision)
    if expected_recall is None:
        assert np.isnan(result["recall"])
    else:
        assert result["recall"] == pytest.approx(expected_recall)


def test_binary_metrics_uses_only_foreground_for_two_channel_masks() -> None:
    label = torch.tensor(
        [[[[1, 1], [1, 1]], [[1, 0], [0, 0]]]],
        dtype=torch.bool,
    )
    prediction = torch.tensor(
        [[[[0, 0], [0, 0]], [[1, 0], [0, 0]]]],
        dtype=torch.bool,
    )
    false_positive_rate, false_negative_rate, iou, dice = inference.metrics_tensor(
        label, prediction
    )

    assert inference.binary_metrics(label, prediction) == pytest.approx(
        {
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "iou": iou,
            "dice": dice,
            "precision": 1.0,
            "recall": 1.0,
        }
    )


def test_hausdorff95_uses_physical_spacing_and_handles_empty_masks() -> None:
    label = torch.zeros((1, 1, 4, 4, 4), dtype=torch.bool)
    prediction = torch.zeros_like(label)
    label[0, 0, 1, 1, 1] = True
    prediction[0, 0, 2, 1, 1] = True
    assert inference.hausdorff95(label, prediction, (2.0, 1.0, 1.0)) == pytest.approx(2.0)
    assert np.isnan(inference.hausdorff95(torch.zeros_like(label), prediction, (1, 1, 1)))
