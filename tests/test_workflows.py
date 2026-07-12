from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
nib = pytest.importorskip("nibabel")

import inference
import train


def _write_case(root: Path, case: str) -> None:
    (root / "imagesTr" / f"{case}__CT.nii.gz").touch()
    (root / "imagesTr" / f"{case}__PT.nii.gz").touch()
    (root / "labelsTr" / f"{case}.nii.gz").touch()


def test_case_pairing_and_split_reject_missing_modalities(tmp_path):
    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "labelsTr").mkdir()
    for index in range(5):
        _write_case(tmp_path, f"CASE{index:03d}")

    records = train.discover_cases(
        tmp_path / "imagesTr", tmp_path / "imagesTr", tmp_path / "labelsTr"
    )
    train_records, validation_records, test_records = train.split_cases(records)
    assert len(records) == 5
    assert len(train_records) == 3
    assert len(validation_records) == 1
    assert len(test_records) == 1
    assert [record["case"] for record in inference.pair_cases(
        tmp_path / "imagesTr", tmp_path / "imagesTr", tmp_path / "labelsTr"
    )] == [f"CASE{index:03d}" for index in range(5)]

    (tmp_path / "imagesTr" / "CASE004__PT.nii.gz").unlink()
    with pytest.raises(ValueError, match="Case mismatch across modalities"):
        train.discover_cases(
            tmp_path / "imagesTr", tmp_path / "imagesTr", tmp_path / "labelsTr"
        )


def test_training_transforms_pad_patches_and_keep_validation_deterministic():
    train_transform, validation_transform = train.build_transforms((16, 16, 8))
    train_names = [type(transform).__name__ for transform in train_transform.transforms]
    validation_names = [type(transform).__name__ for transform in validation_transform.transforms]
    assert "SpatialPadd" in train_names
    assert "RandCropByPosNegLabeld" in train_names
    assert not any(name.startswith("Rand") for name in validation_names)


def test_checkpoint_loaders_accept_legacy_weights_and_reject_partial_training_state(tmp_path):
    source = torch.nn.Linear(2, 1)
    expected = {key: value.detach().clone() for key, value in source.state_dict().items()}

    legacy_path = tmp_path / "legacy.pth"
    torch.save(expected, legacy_path)
    target = torch.nn.Linear(2, 1)
    assert train.load_checkpoint(legacy_path, target, torch.device("cpu")) == {}
    assert all(torch.equal(target.state_dict()[key], value) for key, value in expected.items())

    full_path = tmp_path / "full.ckpt"
    torch.save({"model_state_dict": expected, "epoch": 4}, full_path)
    with pytest.raises(ValueError, match="partial training metadata"):
        train.load_checkpoint(full_path, target, torch.device("cpu"))
    metadata = train.load_checkpoint(
        full_path,
        target,
        torch.device("cpu"),
        resume_mode="weights-only",
    )
    assert metadata == {}

    prefixed_path = tmp_path / "prefixed.pth"
    torch.save({f"module.{key}": value for key, value in expected.items()}, prefixed_path)
    inference.load_checkpoint(target, prefixed_path, torch.device("cpu"))
    assert all(torch.equal(target.state_dict()[key], value) for key, value in expected.items())


def test_inference_predictor_selects_highest_resolution_output():
    class DeepSupervisionModel(torch.nn.Module):
        def forward(self, ct, pet):
            combined = ct + pet
            return combined, combined[..., ::2, ::2, ::2]

    predictor = inference.InferencePredictor(DeepSupervisionModel())
    inputs = torch.randn(1, 2, 8, 8, 8)
    assert torch.equal(predictor(inputs), inputs[:, :1] + inputs[:, 1:2])


def test_prediction_export_preserves_reference_geometry(tmp_path):
    affine = np.array(
        [[2.0, 0.0, 0.0, 10.0], [0.0, 3.0, 0.0, 20.0],
         [0.0, 0.0, 4.0, 30.0], [0.0, 0.0, 0.0, 1.0]]
    )
    reference = tmp_path / "reference.nii.gz"
    output = tmp_path / "prediction.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), affine), reference)

    inference.save_prediction(torch.ones((4, 4, 4)), reference, output)
    saved = nib.load(output)
    assert np.array_equal(saved.affine, affine)
    assert saved.get_data_dtype() == np.dtype(np.uint8)


def test_cli_parsers_expose_required_paths_and_validation_controls():
    training = train.parse_args([
        "--ct-dir", "ct", "--pet-dir", "pet", "--label-dir", "labels",
        "--val-sw-batch-size", "2", "--val-overlap", "0.25",
    ])
    assert training.val_sw_batch_size == 2
    assert training.val_overlap == pytest.approx(0.25)

    parsed = inference.build_parser().parse_args([
        "--ct-dir", "ct", "--pet-dir", "pet", "--checkpoint", "model.pth"
    ])
    assert parsed.checkpoint == Path("model.pth")
