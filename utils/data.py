"""Shared dataset discovery and spatial validation for paired PET/CT data.

The functions in this module deliberately do not import MONAI.  Filename and
manifest validation can therefore be used by lightweight tooling, while
``ValidateGeometryd`` remains compatible with MONAI dictionary transforms and
MetaTensor objects.
"""

from __future__ import annotations

import csv
from difflib import SequenceMatcher
import random
import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np


NIFTI_SUFFIXES = (".nii.gz", ".nii")
VALID_SPLITS = frozenset({"train", "val", "test"})
_MANIFEST_FIELDS = frozenset({"case", "ct", "pet", "label", "split", "group"})

_MODALITY_PATTERNS = {
    "ct": (r"(?:__|[_-])CT$", r"_0001$"),
    "pet": (r"(?:__|[_-])(?:PT|PET)$", r"_0000$"),
    "label": (r"(?:__|[_-])(?:SEG|GT|LABEL|MASK)$",),
}


def _strip_nifti_suffix(name: str) -> str:
    lower_name = name.casefold()
    for suffix in NIFTI_SUFFIXES:
        if lower_name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Not a NIfTI filename: {name}")


def is_nifti(path: Path) -> bool:
    """Return whether *path* has a case-insensitive ``.nii``/``.nii.gz`` suffix."""
    lower_name = path.name.casefold()
    return any(lower_name.endswith(suffix) for suffix in NIFTI_SUFFIXES)


def case_id(path: Path, modality: str) -> str:
    """Extract a case ID from HECKTOR, nnU-Net, or generic NIfTI names."""
    if modality not in _MODALITY_PATTERNS:
        raise ValueError(f"Unknown modality {modality!r}")
    name = _strip_nifti_suffix(path.name)
    for pattern in _MODALITY_PATTERNS[modality]:
        updated = re.sub(pattern, "", name, flags=re.IGNORECASE)
        if updated != name:
            return updated
    return name


def _validate_case_identifier(case: str, *, context: str) -> None:
    """Require an output-safe, single-component case identifier."""
    unsafe = (
        not case
        or case in {".", ".."}
        or "/" in case
        or "\\" in case
        or any(ord(character) < 32 or ord(character) == 127 for character in case)
    )
    if unsafe:
        raise ValueError(
            f"{context} has unsafe case ID {case!r}; use a non-empty single "
            "filename component without path separators or control characters"
        )


def _split_like_fields(fields: Sequence[str]) -> list[str]:
    suspicious: list[str] = []
    for field in fields:
        if field in _MANIFEST_FIELDS:
            continue
        normalized = re.sub(r"[^a-z0-9]", "", field.casefold())
        if not normalized:
            continue
        similarity = SequenceMatcher(None, normalized, "split").ratio()
        if (
            normalized == "split"
            or normalized.startswith("split")
            or normalized.endswith("split")
            or similarity >= 0.75
        ):
            suspicious.append(field)
    return sorted(suspicious)


def _has_modality_suffix(path: Path, modality: str) -> bool:
    stem = _strip_nifti_suffix(path.name)
    return any(re.search(pattern, stem, re.IGNORECASE) for pattern in _MODALITY_PATTERNS[modality])


def _nifti_files(
    directory: Path, modality: str, *, filter_recognized: bool = False
) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{modality.upper()} directory does not exist: {directory}")
    files = sorted(
        (path for path in directory.iterdir() if path.is_file() and is_nifti(path)),
        key=lambda path: path.name.casefold(),
    )
    if not files:
        raise ValueError(f"No NIfTI files found in {modality.upper()} directory: {directory}")

    # A shared imagesTr directory contains both modalities.  In separate
    # directories generic, identical case names are valid and must be retained.
    if filter_recognized and modality in {"ct", "pet"}:
        recognized = [path for path in files if _has_modality_suffix(path, modality)]
        if recognized:
            return recognized
    return files


def _index_directory(
    directory: Path, modality: str, *, filter_recognized: bool = False
) -> dict[str, tuple[str, Path]]:
    indexed: dict[str, tuple[str, Path]] = {}
    for path in _nifti_files(directory, modality, filter_recognized=filter_recognized):
        display_id = case_id(path, modality)
        if not display_id:
            raise ValueError(f"Empty case ID derived from {modality.upper()} file: {path}")
        _validate_case_identifier(display_id, context=f"{modality.upper()} file {path}")
        key = display_id.casefold()
        if key in indexed:
            raise ValueError(
                f"Duplicate {modality.upper()} case ID {display_id!r}: "
                f"{indexed[key][1]} and {path}"
            )
        indexed[key] = (display_id, path.resolve())
    return indexed


def _pair_directories(
    ct_dir: Path, pet_dir: Path, label_dir: Path | None
) -> list[dict[str, str]]:
    shared_image_directory = ct_dir.resolve() == pet_dir.resolve()
    indexes = {
        "ct": _index_directory(
            ct_dir, "ct", filter_recognized=shared_image_directory
        ),
        "pet": _index_directory(
            pet_dir, "pet", filter_recognized=shared_image_directory
        ),
    }
    if shared_image_directory:
        ct_paths = {entry[1] for entry in indexes["ct"].values()}
        pet_paths = {entry[1] for entry in indexes["pet"].values()}
        if ct_paths & pet_paths:
            raise ValueError(
                "CT and PET use the same directory but filenames do not identify "
                "the modalities; use separate directories or a CSV manifest."
            )
        unrecognized = sorted(
            str(path.resolve())
            for path in ct_dir.iterdir()
            if path.is_file()
            and is_nifti(path)
            and path.resolve() not in ct_paths | pet_paths
        )
        if unrecognized:
            raise ValueError(
                "Shared CT/PET directory contains NIfTI files with no recognized "
                f"modality suffix: {unrecognized}"
            )
    if label_dir is not None:
        indexes["label"] = _index_directory(label_dir, "label")

    expected = set(indexes["ct"])
    problems: list[str] = []
    for modality, index in indexes.items():
        if modality == "ct":
            continue
        missing = sorted(indexes["ct"][key][0] for key in expected - set(index))
        extra = sorted(index[key][0] for key in set(index) - expected)
        if missing or extra:
            problems.append(f"{modality.upper()}: missing={missing}, extra={extra}")
    if problems:
        raise ValueError("Case mismatch across modalities; " + "; ".join(problems))

    records: list[dict[str, str]] = []
    for key in sorted(expected):
        record = {
            "case": indexes["ct"][key][0],
            "ct": str(indexes["ct"][key][1]),
            "pet": str(indexes["pet"][key][1]),
        }
        if "label" in indexes:
            record["label"] = str(indexes["label"][key][1])
        records.append(record)
    return records


def _resolve_manifest_path(value: str, manifest: Path, field: str, case: str) -> Path:
    if not value.strip():
        raise ValueError(f"Manifest case {case!r} has an empty {field!r} path")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest case {case!r} {field} file does not exist: {path}")
    if not is_nifti(path):
        raise ValueError(f"Manifest case {case!r} {field} is not a NIfTI file: {path}")
    return path


def read_manifest(manifest: Path, require_labels: bool = False) -> list[dict[str, str]]:
    """Read a canonical ``case,ct,pet[,label,split,group]`` CSV manifest.

    Relative image paths are resolved against the manifest's directory.  If
    ``label`` is used, or a ``split``/``group`` header is present, that field
    must be populated for every row.
    """
    manifest = manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"case", "ct", "pet"}
        missing_fields = sorted(required - fields)
        if missing_fields:
            raise ValueError(f"Manifest is missing required columns: {', '.join(missing_fields)}")
        suspicious_fields = _split_like_fields(tuple(fields))
        if suspicious_fields:
            raise ValueError(
                "Manifest has an unrecognized split-like column "
                f"{', '.join(repr(field) for field in suspicious_fields)}; "
                "use the exact 'split' header"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest contains no cases: {manifest}")

    label_used = "label" in fields and any((row.get("label") or "").strip() for row in rows)
    split_used = "split" in fields
    group_used = "group" in fields
    if require_labels and not label_used:
        raise ValueError("Training/evaluation requires a populated manifest 'label' column")

    records: list[dict[str, str]] = []
    seen_cases: dict[str, int] = {}
    seen_paths: dict[tuple[str, Path], str] = {}
    path_owners: dict[Path, tuple[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        case = (row.get("case") or "").strip()
        if not case:
            raise ValueError(f"Manifest line {line_number} has an empty case ID")
        _validate_case_identifier(case, context=f"Manifest line {line_number}")
        normalized_case = case.casefold()
        if normalized_case in seen_cases:
            raise ValueError(
                f"Duplicate manifest case ID {case!r} on lines "
                f"{seen_cases[normalized_case]} and {line_number}"
            )
        seen_cases[normalized_case] = line_number

        record = {"case": case}
        for modality in ("ct", "pet"):
            path = _resolve_manifest_path(row.get(modality) or "", manifest, modality, case)
            owner = seen_paths.get((modality, path))
            if owner is not None:
                raise ValueError(
                    f"Manifest reuses {modality.upper()} file {path} for "
                    f"cases {owner!r} and {case!r}"
                )
            other_owner = path_owners.get(path)
            if other_owner is not None:
                raise ValueError(
                    f"Manifest file {path} is assigned to both "
                    f"{other_owner[0].upper()} case {other_owner[1]!r} and "
                    f"{modality.upper()} case {case!r}"
                )
            seen_paths[(modality, path)] = case
            path_owners[path] = (modality, case)
            record[modality] = str(path)

        if label_used:
            path = _resolve_manifest_path(row.get("label") or "", manifest, "label", case)
            owner = seen_paths.get(("label", path))
            if owner is not None:
                raise ValueError(
                    f"Manifest reuses label file {path} for cases "
                    f"{owner!r} and {case!r}"
                )
            other_owner = path_owners.get(path)
            if other_owner is not None:
                raise ValueError(
                    f"Manifest file {path} is assigned to both "
                    f"{other_owner[0].upper()} case {other_owner[1]!r} and "
                    f"LABEL case {case!r}"
                )
            seen_paths[("label", path)] = case
            path_owners[path] = ("label", case)
            record["label"] = str(path)
        if split_used:
            split = (row.get("split") or "").strip().casefold()
            if not split:
                raise ValueError(
                    f"Manifest line {line_number} has an empty 'split' value; "
                    "the column must be populated for every case"
                )
            if split not in VALID_SPLITS:
                raise ValueError(
                    f"Manifest case {case!r} has invalid split {split!r}; "
                    f"expected one of {sorted(VALID_SPLITS)}"
                )
            record["split"] = split
        if group_used:
            group = (row.get("group") or "").strip()
            if not group:
                raise ValueError(
                    f"Manifest line {line_number} has an empty 'group' value; "
                    "the column must be populated for every case"
                )
            record["group"] = group
        records.append(record)
    return records


def discover_cases(
    ct_dir: Path | None = None,
    pet_dir: Path | None = None,
    label_dir: Path | None = None,
    *,
    manifest: Path | None = None,
    require_labels: bool = False,
) -> list[dict[str, str]]:
    """Discover paired cases from directories or an explicit CSV manifest."""
    directories_supplied = any(path is not None for path in (ct_dir, pet_dir, label_dir))
    if manifest is not None:
        if directories_supplied:
            raise ValueError("Use either --manifest or dataset directories, not both")
        return read_manifest(manifest, require_labels=require_labels)
    if ct_dir is None or pet_dir is None:
        raise ValueError("CT and PET directories are required when no manifest is provided")
    if require_labels and label_dir is None:
        raise ValueError("A label directory is required for training/evaluation")
    return _pair_directories(ct_dir, pet_dir, label_dir)


def split_cases(
    cases: Sequence[Mapping[str, str]],
    *,
    mode: str = "legacy",
    seed: int = 12345,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Resolve manifest splits or a deterministic 60/20/20 split.

    ``legacy`` sorts by case ID. ``seeded`` shuffles that stable ordering using
    a local RNG.  Explicit manifest splits always take precedence.
    """
    if not cases:
        raise ValueError("Cannot split an empty dataset")
    copied = [dict(case) for case in cases]
    for case in copied:
        case_name = case.get("case")
        if not isinstance(case_name, str):
            raise ValueError("Every dataset record must have a string case ID")
        _validate_case_identifier(case_name, context="Dataset record")

    split_flags = ["split" in case for case in copied]
    group_flags = ["group" in case for case in copied]
    if any(group_flags) and not all(group_flags):
        raise ValueError("Group must be specified for every case or for none")
    if any(split_flags):
        if not all(split_flags):
            raise ValueError("Split must be specified for every case or for none")
        partitions = {name: [] for name in VALID_SPLITS}
        group_splits: dict[str, tuple[str, str]] = {}
        for case in copied:
            split = case["split"].casefold()
            if split not in VALID_SPLITS:
                raise ValueError(f"Case {case.get('case')!r} has invalid split {split!r}")
            if group_flags[0]:
                group = case["group"].strip()
                if not group:
                    raise ValueError(f"Case {case.get('case')!r} has an empty group ID")
                normalized_group = group.casefold()
                previous = group_splits.get(normalized_group)
                if previous is not None and previous[0] != split:
                    raise ValueError(
                        f"Dataset group {previous[1]!r} is assigned to multiple splits "
                        f"({previous[0]!r} and {split!r})"
                    )
                group_splits[normalized_group] = (split, group)
            partitions[split].append(case)
        if not partitions["train"] or not partitions["val"]:
            raise ValueError("Explicit splits must contain at least one train and one val case")
        return partitions["train"], partitions["val"], partitions["test"]

    if group_flags[0]:
        group_counts: dict[str, tuple[str, int]] = {}
        for case in copied:
            group = case["group"].strip()
            if not group:
                raise ValueError(f"Case {case.get('case')!r} has an empty group ID")
            normalized_group = group.casefold()
            display, count = group_counts.get(normalized_group, (group, 0))
            group_counts[normalized_group] = (display, count + 1)
        repeated = sorted(display for display, count in group_counts.values() if count > 1)
        if repeated:
            raise ValueError(
                "Repeated group IDs require explicit manifest splits to prevent "
                f"data leakage: {repeated}"
            )

    ordered = sorted(copied, key=lambda case: case["case"].casefold())
    if mode == "seeded":
        random.Random(seed).shuffle(ordered)
    elif mode != "legacy":
        raise ValueError("split mode must be 'legacy' or 'seeded'")
    train_end = int(0.6 * len(ordered))
    val_end = int(0.8 * len(ordered))
    train, val, test = ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]
    if not train or not val or not test:
        raise ValueError(
            f"Dataset is too small for a non-empty 60/20/20 split "
            f"({len(ordered)} cases)"
        )
    return train, val, test


def _spatial_shape(value: Any) -> tuple[int, ...]:
    shape = tuple(int(item) for item in value.shape)
    if len(shape) < 3:
        raise ValueError(f"Expected at least 3 spatial dimensions, got shape {shape}")
    spatial_shape = shape[-3:]
    if any(dimension <= 0 for dimension in spatial_shape):
        raise ValueError(f"Expected positive spatial dimensions, got shape {shape}")
    return spatial_shape


def _array_data(value: Any) -> np.ndarray | None:
    """Return numeric storage when *value* exposes an array/tensor protocol."""
    converted = value
    has_array_storage = isinstance(value, np.ndarray) or hasattr(value, "__array__")
    if hasattr(converted, "detach"):
        converted = converted.detach()
        has_array_storage = True
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        try:
            return np.asarray(converted.numpy())
        except (TypeError, ValueError):
            pass
    if has_array_storage:
        try:
            return np.asarray(converted)
        except (TypeError, ValueError):
            return None
    return None


def foreground_above_minimum(values: Any) -> Any:
    """Picklable PET foreground predicate with fail-fast content checks."""
    array = _array_data(values)
    if array is None:
        raise ValueError("PET crop values must expose numeric array data")
    if array.size == 0:
        raise ValueError("PET crop values are empty")
    if array.dtype.kind not in "buif" or not np.isfinite(array).all():
        raise ValueError("PET crop values contain non-finite or non-real values")
    if not np.any(array > np.min(array)):
        raise ValueError("PET crop values are constant; foreground would be empty")
    return values > values.min()


def _spatial_mask(mask: np.ndarray) -> np.ndarray:
    leading_axes = tuple(range(mask.ndim - 3))
    return np.any(mask, axis=leading_axes) if leading_axes else mask


def _mask_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower: list[int] = []
    upper: list[int] = []
    for axis in range(mask.ndim):
        other_axes = tuple(index for index in range(mask.ndim) if index != axis)
        occupied = np.flatnonzero(np.any(mask, axis=other_axes))
        lower.append(int(occupied[0]))
        upper.append(int(occupied[-1]) + 1)
    return np.asarray(lower), np.asarray(upper)


def _validate_pet_crop_contains_label(
    arrays: Mapping[str, np.ndarray], *, case: str
) -> None:
    pet = arrays.get("pet")
    label = arrays.get("label")
    if pet is None or label is None:
        return
    pet_mask = _spatial_mask(pet > np.min(pet))
    label_mask = _spatial_mask(label != 0)
    if not np.any(label_mask):
        return

    lower, upper = _mask_bounds(pet_mask)
    crop = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    total = int(np.count_nonzero(label_mask))
    retained = int(np.count_nonzero(label_mask[crop]))
    discarded = total - retained
    if discarded:
        raise ValueError(
            f"Invalid training crop for case {case!r}: PET foreground bbox would "
            f"discard {discarded} of {total} nonzero label voxels"
        )


def _affine(data: Mapping[str, Any], key: str) -> np.ndarray:
    value = data[key]
    affine = getattr(value, "affine", None)
    if affine is None:
        metadata = data.get(f"{key}_meta_dict")
        if isinstance(metadata, Mapping):
            affine = metadata.get("affine")
    if affine is None:
        raise ValueError(f"{key!r} has no affine metadata after loading")
    if hasattr(affine, "detach"):
        affine = affine.detach()
    if hasattr(affine, "cpu"):
        affine = affine.cpu()
    result = np.asarray(affine, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{key!r} has an invalid affine with shape {result.shape}")
    return result


def validate_geometry(
    data: Mapping[str, Any],
    *,
    keys: Sequence[str],
    reference_key: str,
    case_key: str = "case",
    validate_pet_crop: bool = False,
    affine_atol: float = 1e-4,
    affine_rtol: float = 1e-5,
) -> None:
    """Fail on unsafe image content/geometry and optional PET training crops."""
    if reference_key not in keys:
        raise ValueError("reference_key must be included in keys")
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Geometry validation is missing keys: {', '.join(missing)}")
    case = str(data.get(case_key, "<unknown>"))
    for key in keys:
        shape = tuple(int(item) for item in data[key].shape)
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError(
                f"Channel mismatch for case {case!r}: {key} must have exactly one "
                f"channel after loading, got shape {shape}"
            )
    reference_shape = _spatial_shape(data[reference_key])
    reference_affine = _affine(data, reference_key)
    for key in keys:
        if key == reference_key:
            continue
        shape = _spatial_shape(data[key])
        if shape != reference_shape:
            raise ValueError(
                f"Geometry mismatch for case {case!r}: {key} spatial shape {shape} "
                f"!= {reference_key} spatial shape {reference_shape}"
            )
        affine = _affine(data, key)
        if not np.allclose(affine, reference_affine, atol=affine_atol, rtol=affine_rtol):
            max_difference = float(np.max(np.abs(affine - reference_affine)))
            raise ValueError(
                f"Geometry mismatch for case {case!r}: {key} affine differs from "
                f"{reference_key} (max absolute difference={max_difference:.6g})"
            )

    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        array = _array_data(data[key])
        if array is None:
            continue
        expected_shape = tuple(int(item) for item in data[key].shape)
        if array.shape != expected_shape:
            raise ValueError(
                f"Invalid volume for case {case!r}: {key} exposes array shape "
                f"{array.shape}, expected {expected_shape}"
            )
        if array.dtype.kind not in "buif":
            raise ValueError(
                f"Invalid volume for case {case!r}: {key} must have a real "
                f"numeric dtype, got {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"Invalid volume for case {case!r}: {key} contains non-finite values"
            )
        arrays[key] = array

    if validate_pet_crop:
        pet = arrays.get("pet")
        if pet is not None and not np.any(pet > np.min(pet)):
            raise ValueError(
                f"Invalid volume for case {case!r}: PET is constant; its foreground "
                "crop would be empty"
            )
        _validate_pet_crop_contains_label(arrays, case=case)


class ValidateGeometryd:
    """MONAI-compatible content/geometry validator with opt-in crop checks."""

    def __init__(
        self,
        keys: Sequence[str],
        reference_key: str,
        *,
        case_key: str = "case",
        validate_pet_crop: bool = False,
        affine_atol: float = 1e-4,
        affine_rtol: float = 1e-5,
    ) -> None:
        self.keys = tuple(keys)
        self.reference_key = reference_key
        self.case_key = case_key
        self.validate_pet_crop = validate_pet_crop
        self.affine_atol = affine_atol
        self.affine_rtol = affine_rtol

    def __call__(self, data: Mapping[str, Any]) -> MutableMapping[str, Any]:
        result = dict(data)
        validate_geometry(
            result,
            keys=self.keys,
            reference_key=self.reference_key,
            case_key=self.case_key,
            validate_pet_crop=self.validate_pet_crop,
            affine_atol=self.affine_atol,
            affine_rtol=self.affine_rtol,
        )
        return result
