# Preprocessing and input contract

This document separates the maintained input contract from historical preprocessing evidence. The maintained training and inference entry points validate prepared NIfTI files; they do not normalize, register, or resample them.

## Maintained input contract

For each case:

- CT and PET are single-channel 3D NIfTI volumes.
- A label is required for training and optional for inference.
- CT is the reference grid.
- PET and label must match the CT spatial shape and voxel-to-world affine within the implemented numerical tolerance.
- Non-zero label values are foreground in a binary segmentation task.
- Intensities must already follow a consistent, documented convention appropriate for the chosen checkpoint or experiment.

The affine comparison jointly detects differences in voxel spacing, axis orientation, and physical placement. A mismatch raises an error before tensors are concatenated. H2ASeg deliberately does not guess which image is correct and does not automatically resample any modality.

For a new dataset, prepare all modalities outside this repository, inspect the registered volumes visually, and record the full preprocessing pipeline with the experiment. At minimum, document source modality units, conversion, registration reference and transform, resampling grid and interpolation, intensity clipping/normalization, and label definition. The H2ASeg repository does not provide a universal PET/CT normalization prescription.

## Historical evidence from `origin/master`

The old `preprocessing.py` and `registration.py` files in the historical `origin/master` branch reveal an earlier local workflow. They are evidence of implementation history, not a validated turnkey pipeline: both contain hard-coded private paths, omit provenance for several runtime values, and are not part of the maintained entry points.

### AutoPET-II intensity handling

The historical `preprocessing.py`:

1. loaded nnU-Net-style AutoPET-II CT (`_0001`), PET (`_0000`), and label volumes;
2. collected CT intensities from voxels where the tumour label was greater than zero across the dataset;
3. calculated dataset-level mean, standard deviation, and the 0.5th/99.5th percentiles from those labelled CT voxels;
4. clipped every CT volume to those percentile bounds, then standardized it using that labelled-voxel mean and standard deviation; and
5. wrote images while copying origin, direction, and spacing metadata from the source image.

The PET call passes `pet_data > pet_data.min()` as the intended foreground mask for per-volume z-score normalization. However, the historical `zscore` function then computes `mask = seg >= 0`. Because the supplied `seg` is boolean, this selects both `False` and `True` entries and therefore standardizes all voxels, not only values above the minimum. Reimplementations must not silently choose between the intended and actual behavior: state the choice and validate it for the checkpoint being evaluated.

The script also truncated a mismatched label array by index to the CT array shape. The maintained code rejects shape or affine mismatches instead, because truncation does not establish physical alignment.

No modality unit (for example, a specific PET SUV representation) is asserted by these files. Do not infer units from filenames or normalization code.

### HECKTOR registration evidence

The historical `registration.py`:

- used CT as the fixed ANTs image and PET as the moving image;
- requested an affine registration;
- applied the resulting forward transform to the label using nearest-neighbour interpolation; and
- configured a nominal CT spacing of `(1, 1, 1)` before registration.

The script calls `ants.set_spacing` on the CT image rather than documenting a complete resampling operation, and it does not preserve a machine-readable preprocessing manifest. Treat `(1, 1, 1)` as a historical script setting, not as a universally appropriate spacing or proof that every saved output shares a correct physical grid.

## Recommended preparation checklist

Before training or inference on a new dataset:

1. Establish the physical meaning and units of CT and PET intensities from the dataset documentation.
2. Select and document the reference space, registration method, transform direction, target spacing, and interpolation per modality. Use nearest-neighbour interpolation for segmentation labels.
3. Inspect multimodal overlays and labels; affine equality alone cannot prove registration quality.
4. Define intensity preprocessing from evidence appropriate to the dataset and checkpoint. Apply it consistently to train, validation, and test cases without leaking test statistics.
5. Use an explicit CSV manifest for official patient/centre splits. Do not infer patient grouping or centre isolation from arbitrary filenames.
6. Run the repository geometry checks and record the resolved split alongside the experiment.
7. Compare validation and standalone inference using the same checkpoint, overlap, and sliding-window configuration.

## What geometry validation cannot guarantee

Matching shape and affine means arrays occupy the same declared voxel grid. It does not prove that registration is anatomically correct, that headers are truthful, that intensity units are comparable, or that a dataset is suitable for the original model. Those properties require dataset knowledge, visual quality assurance, and external validation.
