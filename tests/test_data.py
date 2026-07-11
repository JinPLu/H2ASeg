import csv
from pathlib import Path

import numpy as np
import pytest

from utils.data import (
    ValidateGeometryd,
    case_id,
    discover_cases,
    foreground_above_minimum,
    read_manifest,
    split_cases,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_case_id_supports_common_names_case_insensitively():
    assert case_id(Path("CHB-001__CT.NII.GZ"), "ct") == "CHB-001"
    assert case_id(Path("CHB-001_PT.nii"), "pet") == "CHB-001"
    assert case_id(Path("CHB-001_0001.nii.gz"), "ct") == "CHB-001"
    assert case_id(Path("CHB-001_0000.nii.gz"), "pet") == "CHB-001"
    assert case_id(Path("CHB-001_MASK.nii.gz"), "label") == "CHB-001"


def test_directory_discovery_supports_shared_hecktor_and_nnunet_layouts(tmp_path):
    images = tmp_path / "imagesTr"
    labels = tmp_path / "labelsTr"
    for case in ("CASE001", "CASE002"):
        _touch(images / f"{case}__CT.nii.gz")
        _touch(images / f"{case}__PT.nii.gz")
        _touch(labels / f"{case}_SEG.nii.gz")
    records = discover_cases(images, images, labels, require_labels=True)
    assert [record["case"] for record in records] == ["CASE001", "CASE002"]

    nnunet = tmp_path / "nnunet"
    nnlabels = tmp_path / "nnlabels"
    _touch(nnunet / "PATIENT_0000.nii.gz")
    _touch(nnunet / "PATIENT_0001.nii.gz")
    _touch(nnlabels / "PATIENT.nii.gz")
    assert discover_cases(nnunet, nnunet, nnlabels)[0]["case"] == "PATIENT"


def test_directory_discovery_supports_generic_separate_directories(tmp_path):
    ct, pet = tmp_path / "ct", tmp_path / "pet"
    _touch(ct / "subject-A.NII")
    _touch(pet / "subject-A.NII")
    records = discover_cases(ct, pet)
    assert records[0]["case"] == "subject-A"
    assert "label" not in records[0]


def test_separate_directories_do_not_silently_ignore_generic_files(tmp_path):
    ct, pet = tmp_path / "ct", tmp_path / "pet"
    _touch(ct / "A_CT.nii.gz")
    _touch(ct / "UNEXPECTED.nii.gz")
    _touch(pet / "A_PT.nii.gz")
    with pytest.raises(ValueError, match="UNEXPECTED"):
        discover_cases(ct, pet)


def test_directory_discovery_rejects_duplicate_missing_and_extra_cases(tmp_path):
    ct, pet = tmp_path / "ct", tmp_path / "pet"
    _touch(ct / "A_CT.nii.gz")
    _touch(ct / "a__CT.nii")
    _touch(pet / "A_PT.nii.gz")
    with pytest.raises(ValueError, match="Duplicate CT case ID"):
        discover_cases(ct, pet)

    (ct / "a__CT.nii").unlink()
    _touch(pet / "EXTRA_PT.nii.gz")
    with pytest.raises(ValueError, match=r"missing=.*extra=.*EXTRA"):
        discover_cases(ct, pet)


def test_same_generic_directory_is_rejected_as_ambiguous(tmp_path):
    images = tmp_path / "images"
    _touch(images / "CASE001.nii.gz")
    with pytest.raises(ValueError, match="same directory"):
        discover_cases(images, images)


def test_shared_directory_rejects_unrecognized_extra_nifti(tmp_path):
    images = tmp_path / "images"
    _touch(images / "CASE001_CT.nii.gz")
    _touch(images / "CASE001_PT.nii.gz")
    _touch(images / "UNRECOGNIZED.nii.gz")
    with pytest.raises(ValueError, match="no recognized modality suffix"):
        discover_cases(images, images)


def test_manifest_resolves_relative_paths_and_validates_splits(tmp_path):
    for name in ("ct/A.nii.gz", "pet/A.nii.gz", "labels/A.nii.gz"):
        _touch(tmp_path / name)
    manifest = tmp_path / "dataset.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "ct", "pet", "label", "split"])
        writer.writeheader()
        writer.writerow({
            "case": "A", "ct": "ct/A.nii.gz", "pet": "pet/A.nii.gz",
            "label": "labels/A.nii.gz", "split": "train",
        })
    record = read_manifest(manifest, require_labels=True)[0]
    assert record["ct"] == str((tmp_path / "ct/A.nii.gz").resolve())
    assert record["split"] == "train"

    text = manifest.read_text().replace("train", "development")
    manifest.write_text(text)
    with pytest.raises(ValueError, match="invalid split"):
        read_manifest(manifest)


def test_manifest_rejects_partial_optional_columns_and_mixed_inputs(tmp_path):
    for name in ("A_CT.nii.gz", "A_PT.nii.gz", "B_CT.nii.gz", "B_PT.nii.gz"):
        _touch(tmp_path / name)
    manifest = tmp_path / "dataset.csv"
    manifest.write_text(
        "case,ct,pet,label,split\n"
        "A,A_CT.nii.gz,A_PT.nii.gz,,train\n"
        "B,B_CT.nii.gz,B_PT.nii.gz,A_CT.nii.gz,\n"
    )
    with pytest.raises(ValueError, match="empty 'label' path"):
        read_manifest(manifest)
    with pytest.raises(ValueError, match="either --manifest"):
        discover_cases(tmp_path, tmp_path, manifest=manifest)


def test_legacy_seeded_and_manifest_splits_are_deterministic():
    cases = [{"case": f"CASE{index:02d}", "ct": "x", "pet": "y"} for index in range(10)]
    legacy = split_cases(cases)
    seeded_a = split_cases(cases, mode="seeded", seed=7)
    seeded_b = split_cases(cases, mode="seeded", seed=7)
    assert [len(part) for part in legacy] == [6, 2, 2]
    assert seeded_a == seeded_b
    assert seeded_a != legacy

    explicit = [
        {"case": "A", "ct": "x", "pet": "y", "split": "train"},
        {"case": "B", "ct": "x", "pet": "y", "split": "val"},
        {"case": "C", "ct": "x", "pet": "y", "split": "test"},
    ]
    resolved = [[item["case"] for item in part] for part in split_cases(explicit)]
    assert resolved == [["A"], ["B"], ["C"]]


class _Image:
    def __init__(self, shape, affine):
        self.shape = shape
        self.affine = np.asarray(affine)


def test_geometry_validator_accepts_aligned_modalities_and_is_non_mutating():
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    data = {
        "case": "A",
        "ct": _Image((1, 8, 9, 10), affine),
        "pet": _Image((1, 8, 9, 10), affine.copy()),
        "label": _Image((1, 8, 9, 10), affine.copy()),
    }
    result = ValidateGeometryd(("ct", "pet", "label"), "ct")(data)
    assert result is not data
    assert result["ct"] is data["ct"]


def test_geometry_validator_rejects_shape_and_affine_mismatch():
    identity = np.eye(4)
    base = {"case": "CASE-X", "ct": _Image((1, 8, 8, 8), identity)}
    with pytest.raises(ValueError, match="CASE-X.*spatial shape"):
        ValidateGeometryd(("ct", "pet"), "ct")({
            **base, "pet": _Image((1, 8, 8, 7), identity),
        })

    shifted = identity.copy()
    shifted[0, 3] = 5.0
    with pytest.raises(ValueError, match="CASE-X.*affine differs"):
        ValidateGeometryd(("ct", "pet"), "ct")({
            **base, "pet": _Image((1, 8, 8, 8), shifted),
        })


def test_geometry_validator_rejects_multichannel_inputs():
    identity = np.eye(4)
    with pytest.raises(ValueError, match="must have exactly one channel"):
        ValidateGeometryd(("ct", "pet"), "ct")({
            "case": "MULTICHANNEL",
            "ct": _Image((2, 8, 8, 8), identity),
            "pet": _Image((1, 8, 8, 8), identity),
        })


def test_foreground_predicate_is_module_level_and_preserves_array_semantics():
    values = np.array([0, 0, 2])
    assert foreground_above_minimum(values).tolist() == [False, False, True]
