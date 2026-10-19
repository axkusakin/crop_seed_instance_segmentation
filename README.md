# Crop Seed Instance Segmentation

Mask R-CNN-based instance segmentation of crop seeds (barley, plus rice and wheat data/weights from the original
project), with scripts to turn detections into per-seed morphological measurements and summary tables.

This is a fork of the original research code (see [Background](#background) below) that:
* Vendors and patches Mask R-CNN (`mrcnn/`) directly into this repository — no separate clone step needed. See
  [`mrcnn/PATCHES.md`](./mrcnn/PATCHES.md) for exactly what was changed and why.
* Updates the installation process for Python 3.11 / TensorFlow 2.15.

## Setup

**1. Clone the repository:**
```
git clone https://github.com/axkusakin/crop_seed_instance_segmentation.git
cd crop_seed_instance_segmentation
```

**2. Create the environment (Python 3.11 required):**
```
mamba create -n <env_name> python=3.11
conda activate <env_name>
```

**3. Install dependencies:**
```
pip install -r requirements.txt
```

`mrcnn/` (Mask R-CNN) is vendored in this repository as a patched fork of
[ahmedfgad/Mask-RCNN-TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2) (itself a TF2 port of
[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)), patched to run under modern `tf.keras`
(TensorFlow 2.15). It imports directly as `mrcnn.model`, `mrcnn.config`, etc. as long as you run scripts/notebooks
from the repository root — no separate clone or `python setup.py install` step is needed.

**Verified working combination:** Python 3.11, TensorFlow 2.15.0, Keras 2.15.0, NumPy 1.26.4. NumPy must stay below
2.0 — TensorFlow 2.15 was built against the NumPy 1.x ABI and fails to import under NumPy 2.x. If you add or upgrade
dependencies, re-verify that `numpy==1.26.4` is still what gets installed.

Only the inference path (`mrcnn.model.MaskRCNN(mode="inference")`, `.detect()`, `.load_weights()`) has been patched
and tested, matching how the scripts below and `Mask_RCNN.ipynb` use it. Training-only code paths (`.compile()`,
`.train()`) were not exercised or fixed and should be assumed broken until someone needs them.

## Getting the data and pretrained weights

```
python data_downloader.py [--force]
```

Downloads `data.zip` from Google Drive, extracts it into `data/` (gitignored), and cleans up. `--force` overwrites
an existing `data.zip`/`data/`. Contents:
* Barley: synthetic images/masks (test data), real-world images of 19 cultivars with JSON annotations, and trained
  model weights.
* Other crops: model weights and images for rice, and for 4 wheat cultivars (one model infers all 4).

## Extracting seed morphology metrics

`grain_metrics_and_visualize.py` detects grains with Mask R-CNN and computes per-seed morphological metrics. By
default every seed detected in an image is treated as belonging to one accession (`file_name`). With
`-r / --replicates`, the seeds of each image are additionally split into individual-plant replicates, using the image
file name plus the spatial gaps between plants (see [Replicate file-name convention](#replicate-file-name-convention)).

```
# Whole-image accession analysis
python grain_metrics_and_visualize.py -i images -o output_dir [--save-images] [--dpi 500]

# Same, but split into per-plant replicates
python grain_metrics_and_visualize.py -i images -o output_dir --replicates [--save-images] [--dpi 500]
```

Supported input formats: TIF, TIFF, PNG, JPEG.

All parameters can also be kept in a YAML config file — copy [`config.example.yaml`](./config.example.yaml) and pass
it with `-c / --config`. Flags given on the command line override values from the file (boolean options can be
switched off with `--no-replicates` / `--no-save-images`), and the resolved parameters of every run are saved to
`<output-dir>/config_used.yaml`, so any run can be reproduced with `--config <output-dir>/config_used.yaml`.

```
python grain_metrics_and_visualize.py --config my_config.yaml
python grain_metrics_and_visualize.py --config my_config.yaml --mask-threshold 0.3 -o other_output
```

The script is a thin entry point; the pipeline itself lives in the [`seedseg/`](./seedseg) package (preprocessing,
measurement, replicate assignment, unit conversion, summaries, overlays). Only `seedseg/model.py` imports TensorFlow.

Key flags (run the script with `-h` for the full list):

| Flag | Default | Meaning |
| --- | --- | --- |
| `-c / --config` | none | YAML config file; command-line flags override it |
| `-i / --input` | required | Directory of input images (flag or config) |
| `-o / --output-dir` | required | Root directory for all outputs (flag or config) |
| `-w / --weights` | `data/barley/model_weights/mask_rcnn_barleyseeds_0040.h5` | Mask R-CNN weights path |
| `-r / --replicates` | off | Split each image's seeds into plant replicates |
| `--save-images` | off | Also save prediction overlays |
| `--dpi` | none (pixels) | Scan resolution; converts length/area columns to millimetres |
| `-cl / --crop-left`, `-cr / --crop-right` | `0.0` | Fraction of the image width trimmed from that side (e.g. scanner artifacts) |
| `-ct / --crop-top`, `-cb / --crop-bottom` | `0.0` | Fraction of the image height trimmed from that side |
| `--min-score` | `0.95` | Minimum detection confidence kept in the output table |
| `--min-contour-points` | `100` | Minimum contour point count kept in the output table |
| `--mask-threshold` | `0.1` | Probability threshold for binarising soft masks; lower values widen the masks (`0.5` = original Mask R-CNN behaviour) |
| `--max-instances` | `400` | Maximum Mask R-CNN detections per image |
| `--alpha` | `0.35` | Overlay opacity for `--save-images` |

No IQR or other outlier filtering is applied — every detection that passes `--min-score` and `--min-contour-points`
is written to the table.

### Units

By default all length/area measurements are in pixels. Pass `--dpi` (the scan resolution) to convert them to
millimetres (`mm_per_pixel = 25.4 / dpi`); when set, the output tables report only the millimetre-based columns
(suffixed `_mm`/`_mm2`) plus the dimensionless ratios (length-to-width ratio, eccentricity, solidity, circularity) —
pixel columns are dropped.

### Replicate file-name convention

With `--replicates`, the script expects input file names of the form
`<sample_id>_<r1-r2-...>.ext`, where the trailing underscore-separated token is one or more dash-separated
replicate numbers:

| File name | sample_id | replicate IDs |
| --- | --- | --- |
| `MyCollection1_18-19_015_1-2-3.jpg` | `MyCollection1_18-19_015` | `[1, 2, 3]` |
| `MyCollection_022_5-4-8.jpg` | `MyCollection_022` | `[5, 4, 8]` |
| `MyCollection_007_4.jpg` | `MyCollection_007` | `[4]` (single plant) |

**Lay the plants out vertically.** The seeds of each plant must form a separate group, with the groups placed one
above another and separated by an empty gap. The groups are matched to the replicate numbers from top to bottom:
in `..._5-4-8.jpg` the top group is replicate 5, the middle one 4 and the bottom one 8.

A file that doesn't match this pattern still contributes to `seed_parameters.tsv`, but with `sample_id`/
`replicate_id` left blank and a warning in `run_log.txt`; it's excluded from the replicate/sample summary tables.

### Output

Everything is written under the given `--output-dir`:

```
<output-dir>/
    seed_parameters.tsv     one row per detected seed
    samples_summary.tsv     summary stats (mean/sd/min/max/median) per accession or sample_id
    replicates_summary.tsv  summary stats per (sample_id, replicate_id) -- --replicates only
    run_log.txt             run parameters and a per-run summary
    config_used.yaml        resolved parameters of the run (reusable with --config)
    predicted_masks/        overlay PNGs, only created if --save-images is passed
```

Per-seed metrics: seed area, length, width, length-to-width ratio, eccentricity, solidity, perimeter length,
circularity (plus centroid position with `--replicates`, used internally for replicate assignment).

## Other notebooks

* [`Mask_RCNN.ipynb`](./Mask_RCNN.ipynb) — the original Mask R-CNN inference notebook.
* [`multivariate_analysis.ipynb`](multivariate_analysis.ipynb) — PCA and elliptic Fourier descriptor (EFD) analysis
  of previously extracted phenotypes. It loads *precomputed* EFD coefficient files
  (`data/barley/extracted_phenotypes/efd/*.csv`) rather than computing them from the current scripts' output.

## Background

Original research code and dataset for:

**Learning from Synthetic Dataset for Crop Seed Instance Segmentation.**
See https://www.biorxiv.org/content/10.1101/866921v2 for details.

![Overview of the proposed training process](README.assets/image-20191204160204190.png)

Large files (synthetic/real images and masks, JSON annotations, trained weights) are hosted on
[Google Drive](https://drive.google.com/file/d/1g8bg9ter9DlKWgs0lfPZMQemRlzRVOQr/view?usp=sharing) and fetched by
`data_downloader.py` (see [Getting the data and pretrained weights](#getting-the-data-and-pretrained-weights)).

Original author: Yosuke Toda (JST PRESTO / ITbM, Nagoya University).
