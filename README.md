# [MICCAI 2024] H2ASeg

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Hierarchical Adaptive Interaction and Weighting Network for Tumor Segmentation in PET/CT Images**

> [!IMPORTANT]
> **News**
>
> - **2024 — [H2ASeg](https://doi.org/10.1007/978-3-031-72111-3_30) was published at MICCAI 2024.**
> - **2026 — VeloxSeg was accepted at ICLR 2026.**
>
> H2ASeg records our MICCAI 2024 work on hierarchical PET/CT feature interaction. Our newer project, **[VeloxSeg](https://github.com/JinPLu/VeloxSeg)**, advances this direction toward a more general and efficient segmentation framework designed with clinical deployment in mind. For new research and active development, please start with VeloxSeg.

H2ASeg is the research implementation of a two-stream 3D PET/CT tumor segmentation network. The maintained `main` branch preserves the original model architecture and deep-supervision objective while making data validation, training, inference, checkpoint handling, and device selection easier to reproduce.

> [!WARNING]
> H2ASeg is research code, not a clinically validated medical device.

## Supported scope

- Paired, already registered, single-channel CT and PET NIfTI volumes.
- Binary voxel labels; every non-zero label value is treated as foreground.
- Label-free inference or per-case evaluation when labels are supplied.
- Existing model-only `.pth` state dictionaries and the maintained full-checkpoint format.
- CPU, one selected CUDA device, and CUDA automatic mixed precision for training.

The repository does **not** perform DICOM conversion, registration, resampling, modality-specific intensity normalization, multi-class segmentation, or multi-GPU training. Different datasets are supported only after they have been converted to the input contract below. See [Preprocessing and input contract](docs/preprocessing.md) before comparing results across datasets.

## Installation

Python 3.10 or newer is recommended. The project does not depend on a particular server, cluster, local path, or environment module. On a workstation or server, create an isolated environment and install a PyTorch build appropriate for that machine first:

```bash
git clone https://github.com/JinPLu/H2ASeg.git
cd H2ASeg

python -m venv .venv
source .venv/bin/activate

# Select the appropriate command at https://pytorch.org/get-started/locally/
python -m pip install torch
python -m pip install -r requirements.txt
```

For repository tests, also install:

```bash
python -m pip install -r requirements-dev.txt
```

No private server configuration is required by H2ASeg. Remote execution only needs a checkout, compatible Python/PyTorch environment, prepared data, and enough memory for the selected batch and sliding-window settings.

## Data input

### Directory discovery

Training and inference share the same case-discovery rules. The following layouts are supported.

HECKTOR-style modalities in one image directory:

```text
dataset/
├── imagesTr/
│   ├── CASE001_CT.nii.gz
│   ├── CASE001_PT.nii.gz
│   ├── CASE002_CT.nii.gz
│   └── CASE002_PT.nii.gz
└── labelsTr/
    ├── CASE001.nii.gz
    └── CASE002.nii.gz
```

nnU-Net-style modalities in one image directory are also recognized: `_0001` is CT and `_0000` is PET. Alternatively, put CT and PET in separate directories and use the same generic filename for each case. `.nii` and `.nii.gz` are accepted case-insensitively. Common label suffixes such as `_SEG`, `_GT`, `_LABEL`, and `_MASK` are removed when matching case IDs.

Missing, extra, duplicate, or ambiguous modalities fail with a case-level error. A shared CT/PET directory requires recognizable modality suffixes; use separate directories or a manifest for generic names.

### CSV manifest

A manifest avoids filename inference and can also define an official dataset split:

```csv
case,ct,pet,label,split
CASE001,data/ct/CASE001.nii.gz,data/pet/CASE001.nii.gz,data/labels/CASE001.nii.gz,train
CASE002,data/ct/CASE002.nii.gz,data/pet/CASE002.nii.gz,data/labels/CASE002.nii.gz,val
CASE003,data/ct/CASE003.nii.gz,data/pet/CASE003.nii.gz,data/labels/CASE003.nii.gz,test
```

Required columns are `case`, `ct`, and `pet`. `label` is required for training and optional for inference. `split`, when present, must be populated on every row with `train`, `val`, or `test`. Relative paths are resolved from the manifest directory. Use either `--manifest` or directory arguments, not both.

### Geometry contract

CT is the reference grid. After loading, PET and any label must have the same spatial shape and a voxel-to-world affine equal within a small numerical tolerance. This checks voxel size, orientation, and physical placement together. Geometry mismatches fail before cropping, concatenation, or inference; the code never silently resamples data.

## Training

Train from paired directories:

```bash
python train.py \
  --ct-dir dataset/imagesTr \
  --pet-dir dataset/imagesTr \
  --label-dir dataset/labelsTr \
  --output-dir save/h2aseg_run \
  --device auto \
  --amp
```

Or train from an explicit manifest:

```bash
python train.py \
  --manifest dataset/split.csv \
  --output-dir save/h2aseg_run \
  --device cuda:0
```

If the manifest provides `split`, those assignments take precedence. Without explicit assignments, `--split-mode legacy` sorts case IDs and selects 60%/20%/20% for train/validation/test to preserve historical behavior. `--split-mode seeded --split-seed 12345` shuffles the stable case ordering before the same split. The resolved assignments are written to the run directory; the test partition is recorded but not used during optimization.

Resume a maintained full checkpoint, or initialize model weights from a legacy model-only checkpoint:

```bash
python train.py \
  --manifest dataset/split.csv \
  --output-dir save/h2aseg_run \
  --resume save/h2aseg_run/last.ckpt
```

`--amp` is active only on CUDA. Run `python train.py --help` for the complete, version-matched command-line interface.

A legacy raw or wrapped state dictionary restores model weights only. The maintained `last.ckpt` additionally restores the optimizer, scheduler, AMP scaler, epoch/iteration counters, and available Python, NumPy, PyTorch, CUDA, and DataLoader-generator RNG state. It also records the parsed CLI configuration, model configuration, resolved cases, and resolved split.

Training crops PET foreground before random patch sampling to retain the historical sampling behavior. Validation does not crop: it performs deterministic sliding-window inference over the full input volume, matching standalone inference's spatial evaluation domain.

## Inference and evaluation

Run label-free inference and optionally save NIfTI masks:

```bash
python inference.py \
  --ct-dir dataset/imagesTr \
  --pet-dir dataset/imagesTr \
  --checkpoint save/h2aseg_run/val_best.pth \
  --metrics-csv results/predictions.csv \
  --prediction-dir results/masks \
  --device auto
```

In directory mode, add `--label-dir dataset/labelsTr` to report false-positive rate, false-negative rate, IoU, Dice, and spacing-aware HD95. In manifest mode, a populated `label` column enables the same evaluation; omit that column for label-free inference. Saved masks reuse the reference CT affine and header.

Inference processes every row in the supplied manifest; the optional `split` column controls training partitions but does not filter inference. Provide a test-only manifest when evaluating only a held-out partition.

Inference and validation use the same full-volume sliding-window domain. Their metrics are directly comparable when the checkpoint, input data, overlap, and sliding-window parameters are identical.

## Model configuration constraints

The maintained defaults reproduce the existing entry points:

- patch/ROI size: `(128, 128, 64)`
- encoder channels: `(8, 16, 32, 64, 128)`
- attention window: `(8, 8, 4)`
- inputs: one CT channel and one PET channel
- output: two classes for binary segmentation

H2ASeg performs four stride-2 encoder reductions and uses fixed MCSA window partitioning. A non-default ROI must therefore be divisible by 16 and must produce MCSA feature maps divisible by the configured window and pooling sizes. Invalid combinations fail before model construction. Non-default channel/window settings change parameter shapes and generally cannot load a checkpoint trained with the defaults.

When a maintained full checkpoint contains `model_config`, inference uses that configuration by default. Explicit `--roi-size`, `--num-channels`, or `--window-size` values must agree with the stored metadata. A legacy model-only checkpoint has no metadata and therefore uses the MICCAI defaults unless an explicit, valid configuration is supplied.

## Testing

Run the complete suite in an environment containing the runtime and development dependencies:

```bash
python -m pytest -q
python train.py --help
python inference.py --help
python -m compileall -q train.py inference.py model utils tests
```

The tests cover case discovery, manifests and split behavior, geometry fail-fast checks, model-configuration constraints, metrics edge cases, checkpoint compatibility, deterministic transforms, and NIfTI export. CUDA-specific checks are skipped when CUDA is unavailable; a skip is not evidence that GPU execution passed.

## Reproducibility notes

- No intensity normalization is applied by the maintained training or inference entry points. Supply consistently preprocessed images.
- The historical scripts provide evidence about an earlier AutoPET-II/HECKTOR preparation workflow, but contain hard-coded private paths and a consequential PET masking inconsistency. They should not be run blindly. See [docs/preprocessing.md](docs/preprocessing.md).
- Random rotations are specified in radians (45 degrees by default) and labels use nearest-neighbor interpolation.
- Dice and IoU follow the repository's binary metric convention, including the documented empty-mask behavior in `utils/metrics.py`.
- Checkpoints, TensorBoard logs, NIfTI data, and generated predictions are ignored by Git to reduce accidental publication of large or sensitive artifacts.

## Clinical-use boundary

H2ASeg is provided for research and reproducibility. Performance on public challenge data does not establish safety, effectiveness, robustness across scanners or institutions, or regulatory clearance. Any translational evaluation requires appropriate data governance, site-specific validation, qualified human oversight, and applicable institutional and regulatory review. Do not use this software for diagnosis, treatment decisions, or unsupervised clinical care.

## Citation and license

Please cite the [MICCAI 2024 H2ASeg paper](https://doi.org/10.1007/978-3-031-72111-3_30) when using this repository:

```bibtex
@inproceedings{lu2024h2aseg,
  author    = {Lu, Jinpeng and Chen, Jingyun and Cai, Linghan and Jiang, Songhan and Zhang, Yongbing},
  title     = {{H2ASeg}: Hierarchical Adaptive Interaction and Weighting Network for Tumor Segmentation in {PET/CT} Images},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
  series    = {Lecture Notes in Computer Science},
  volume    = {15008},
  pages     = {316--327},
  publisher = {Springer},
  address   = {Cham},
  year      = {2024},
  doi       = {10.1007/978-3-031-72111-3_30},
  url       = {https://doi.org/10.1007/978-3-031-72111-3_30}
}
```

Copyright 2024 H2ASeg Authors. The source code in this repository is released under the [Apache License 2.0](LICENSE). The license applies to the repository software and documentation; it does not grant rights to the MICCAI paper, datasets, pretrained weights, or other third-party materials unless those items explicitly state otherwise.
