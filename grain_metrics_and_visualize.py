#!/usr/bin/env python3
"""Calculate barley-grain metrics, per-accession summary stats, and optional overlays.

No IQR filtering is applied. Every predicted object that passes --min-score and
--min-contour-points is written to the per-seed table and, when --save-images
is used, drawn in the corresponding overlay. The image label is the exact
zero-based object_id in the table, enabling later manual removal of selected
objects.

Metrics are calculated for the whole accession represented by each image --
this script does not split seeds into replicates.

Units
-----
By default, all length/area measurements are in pixels (pixel and pixel^2).
Pass --dpi to convert them to millimetres, assuming every input image was
scanned at that fixed resolution (mm_per_pixel = 25.4 / dpi). When --dpi is
given, the final table reports ONLY millimetre-based columns (suffixed _mm or
_mm2) plus the dimensionless ratios (LWR, eccentricity, solidity,
circularity) -- pixel columns are dropped from the output. Without --dpi, the
table keeps the original pixel-based column names.

Output structure
-----------------
Everything is written under a single --output-dir:

    <output-dir>/
        seed_parameters.tsv    one row per detected seed
        samples_summary.tsv    summary stats per accession (file_name)
        run_log.txt            run parameters and a per-run summary
        predicted_masks/       only created if --save-images is passed

Examples
--------
# Pixels only
python grain_metrics_no_iqr.py -i images -o output_dir

# Convert to millimetres, assuming a 500 DPI scan
python grain_metrics_no_iqr.py -i images -o output_dir --dpi 500

# Tables plus overlays, cropped edges, millimetre output
python grain_metrics_no_iqr.py -i images -o output_dir --save-images \
    --edge-crop 0.08 --dpi 500
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# Disable the Grappler cost-based optimizations that trigger the CropAndResize warning.
tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

try:
    import mrcnn.model as modellib
    from mrcnn.config import Config
except ImportError:
    sys.exit("Error: Mask R-CNN library not found. Run this script from the repository root.")


class InferenceConfig(Config):
    NAME = "seed"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # background + seed
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 8192
    DETECTION_MIN_CONFIDENCE = 0.0
    RPN_NMS_THRESHOLD = 0.4


# Columns as computed internally -- always in pixels at this stage.
COLUMNS = [
    "file_name", "object_id", "detection_score", "AS_seed_area",
    "L_seed_length", "W_seed_width", "LWR_length_to_width_ratio",
    "eccentricity", "solidity", "PL_perimeter_length", "CS_seed_circularity",
]

# Pixel-valued columns and the power of mm_per_pixel needed to convert them:
# length-like columns scale linearly (^1), area-like columns scale by area (^2).
LENGTH_COLUMNS = ["L_seed_length", "W_seed_width", "PL_perimeter_length"]
AREA_COLUMNS = ["AS_seed_area"]

# Dimensionless columns: never converted, never renamed.
RATIO_COLUMNS = [
    "LWR_length_to_width_ratio", "eccentricity", "solidity", "CS_seed_circularity",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PREDICTED_MASKS_DIRNAME = "predicted_masks"
SEED_PARAMETERS_FILENAME = "seed_parameters.tsv"
SAMPLES_SUMMARY_FILENAME = "samples_summary.tsv"
RUN_LOG_FILENAME = "run_log.txt"


# --------------------------------------------------------------------------
# Console/logging helpers
# --------------------------------------------------------------------------

def format_duration(seconds):
    """Format a duration in seconds as e.g. '1h 03m 12s', '3m 05s', or '4.2s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


class RunLog:
    """Collects console-style lines and writes them to a run_log.txt at the end."""

    def __init__(self):
        self.lines = []

    def emit(self, message=""):
        print(message)
        self.lines.append(message)

    def write(self, path):
        with open(path, "w") as handle:
            handle.write("\n".join(self.lines) + "\n")


# --------------------------------------------------------------------------
# Detection helpers
# --------------------------------------------------------------------------

def instance_colour(object_id):
    """Return a reproducible high-contrast BGR colour for an object ID."""
    hue = ((object_id + 1) * 47) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def crop_lateral_edges(image_bgr, edge_crop):
    """Remove edge_crop of the original width from both lateral sides."""
    width = image_bgr.shape[1]
    edge_px = int(round(width * edge_crop))
    if edge_px == 0:
        return image_bgr
    return image_bgr[:, edge_px:width - edge_px]


def detect_and_measure(result, image_name, min_score, min_contour_points):
    """Convert model output into one metric row and one overlay object per instance.

    All length/area values are computed in pixels; unit conversion happens
    later, once, on the assembled table. Returns the metrics DataFrame, the
    overlay instances, and the raw (pre-filter) instance count reported by
    the model, for console diagnostics.
    """
    masks = result.get("masks")
    scores = result.get("scores", [])
    raw_instance_count = 0 if masks is None or masks.ndim != 3 else masks.shape[-1]
    rows = []
    instances = []

    if masks is None or masks.ndim != 3 or masks.shape[-1] == 0:
        return pd.DataFrame(columns=COLUMNS), instances, raw_instance_count

    for object_id in range(masks.shape[-1]):
        score = float(scores[object_id])
        if score < min_score:
            continue

        mask = masks[:, :, object_id].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if len(contour) >= min_contour_points]
        if not contours:
            continue

        # One model instance yields one table row. Use the largest connected
        # component if a mask contains small disconnected fragments.
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        length, width = max(rect[1]), min(rect[1])
        if width <= 0 or len(contour) < 5:
            continue

        ellipse = cv2.fitEllipse(contour)
        major_axis, minor_axis = max(ellipse[1]), min(ellipse[1])
        if major_axis <= 0:
            continue
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        perimeter = cv2.arcLength(contour, True)
        if hull_area <= 0 or perimeter <= 0:
            continue

        rows.append([
            image_name,
            object_id,
            score,
            area,
            length,
            width,
            length / width,
            eccentricity,
            area / hull_area,
            perimeter,
            (4 * np.pi * area) / (perimeter ** 2),
        ])
        instances.append({
            "object_id": object_id,
            "score": score,
            "mask": mask,
            "contour": contour,
        })

    return pd.DataFrame(rows, columns=COLUMNS), instances, raw_instance_count


def draw_predictions(image_bgr, instances, alpha):
    """Draw every retained object, labelled with its exact table object_id."""
    output = image_bgr.copy()
    overlay = output.copy()

    for instance in instances:
        overlay[instance["mask"].astype(bool)] = instance_colour(instance["object_id"])
    output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)

    for instance in instances:
        object_id = instance["object_id"]
        contour = instance["contour"]
        colour = instance_colour(object_id)
        cv2.drawContours(output, [contour], -1, colour, 2, cv2.LINE_AA)

        moments = cv2.moments(contour)
        if moments["m00"]:
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
        else:
            x, y = contour.reshape(-1, 2)[0]

        label = f"{object_id}: {instance['score']:.2f}"
        cv2.putText(output, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)

    summary = f"Objects in table: {len(instances)}"
    cv2.putText(output, summary, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, summary, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return output


# --------------------------------------------------------------------------
# Unit conversion
# --------------------------------------------------------------------------

def convert_to_millimetres(df, dpi):
    """Convert pixel-based columns to millimetres in place of the pixel columns.

    Length-like columns (length, width, perimeter) scale by mm_per_pixel.
    Area-like columns (seed area) scale by mm_per_pixel squared. Ratio
    columns are left untouched. Returns a new DataFrame with pixel columns
    replaced by their _mm / _mm2 equivalents.
    """
    mm_per_pixel = 25.4 / dpi
    converted = df.copy()

    rename_map = {}
    for column in LENGTH_COLUMNS:
        converted[column] = converted[column] * mm_per_pixel
        rename_map[column] = f"{column}_mm"
    for column in AREA_COLUMNS:
        converted[column] = converted[column] * (mm_per_pixel ** 2)
        rename_map[column] = f"{column}_mm2"

    return converted.rename(columns=rename_map)


def metric_columns_for(dpi):
    """Return the shape-metric column names to summarise, given the active unit."""
    if dpi is None:
        return LENGTH_COLUMNS + AREA_COLUMNS + RATIO_COLUMNS
    length_mm = [f"{column}_mm" for column in LENGTH_COLUMNS]
    area_mm2 = [f"{column}_mm2" for column in AREA_COLUMNS]
    return length_mm + area_mm2 + RATIO_COLUMNS


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

def summarise_by_sample(df, metric_columns):
    """Build a per-accession (file_name) summary with counts and metric stats."""
    aggregations = {
        "n_seeds": ("object_id", "count"),
        "mean_detection_score": ("detection_score", "mean"),
    }
    for metric in metric_columns:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_sd"] = (metric, "std")
        aggregations[f"{metric}_min"] = (metric, "min")
        aggregations[f"{metric}_max"] = (metric, "max")
        aggregations[f"{metric}_median"] = (metric, "median")

    return df.groupby("file_name", dropna=False).agg(**aggregations).reset_index()


# --------------------------------------------------------------------------
# Main processing loop
# --------------------------------------------------------------------------

def process_images(args, model, log):
    predicted_dir = os.path.join(args.output_dir, PREDICTED_MASKS_DIRNAME)
    if args.save_images:
        os.makedirs(predicted_dir, exist_ok=True)

    image_files = [
        filename for filename in sorted(os.listdir(args.input))
        if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        sys.exit(f"Error: no supported images found in {args.input}")

    total_images = len(image_files)
    log.emit(f"Found {total_images} image(s) to process in {args.input}")
    log.emit("-" * 78)

    tables = []
    skipped_files = []
    total_seeds_detected = 0
    run_start = time.perf_counter()

    for index, filename in enumerate(image_files, start=1):
        image_start = time.perf_counter()
        image_path = os.path.join(args.input, filename)
        original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original_bgr is None:
            log.emit(f"[{index}/{total_images}] {filename}: could not read file; skipping.")
            skipped_files.append(filename)
            continue

        image_bgr = crop_lateral_edges(original_bgr, args.edge_crop)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = model.detect([image_rgb], verbose=0)[0]

        metrics, instances, raw_instance_count = detect_and_measure(
            result, filename, args.min_score, args.min_contour_points
        )
        if not metrics.empty:
            tables.append(metrics)

        if args.save_images:
            visualisation = draw_predictions(image_bgr, instances, args.alpha)
            output_name = f"{os.path.splitext(filename)[0]}_predicted.png"
            output_path = os.path.join(predicted_dir, output_name)
            if not cv2.imwrite(output_path, visualisation):
                log.emit(f"Warning: could not write {output_path}")

        elapsed = time.perf_counter() - image_start
        total_seeds_detected += len(instances)

        if instances:
            scores = [instance["score"] for instance in instances]
            score_text = f"mean score {np.mean(scores):.2f} (min {np.min(scores):.2f})"
        else:
            score_text = "no seeds retained"

        avg_time_so_far = (time.perf_counter() - run_start) / index
        remaining_images = total_images - index
        eta_text = format_duration(avg_time_so_far * remaining_images) if remaining_images else "0.0s"
        percent = 100 * index / total_images

        height, width = image_bgr.shape[:2]
        log.emit(
            f"[{index}/{total_images}] ({percent:5.1f}%) {filename}: "
            f"{width}x{height}px | raw detections {raw_instance_count} | "
            f"retained {len(instances)} | {score_text} | "
            f"{elapsed:.2f}s | ETA {eta_text}"
        )

    total_elapsed = time.perf_counter() - run_start
    log.emit("-" * 78)

    if not tables:
        sys.exit("Error: no objects passed the selected score and contour filters. Nothing to save.")

    os.makedirs(args.output_dir, exist_ok=True)

    final_df = pd.concat(tables, ignore_index=True)
    final_df = final_df.sort_values(
        by=["file_name", "object_id"], kind="stable"
    ).reset_index(drop=True)

    if args.dpi is not None:
        final_df = convert_to_millimetres(final_df, args.dpi)
        log.emit(f"Converted pixel measurements to millimetres using {args.dpi} DPI.")

    seed_path = os.path.join(args.output_dir, SEED_PARAMETERS_FILENAME)
    final_df.to_csv(seed_path, sep="\t", index=False)
    log.emit(f"Per-seed table saved to {seed_path}")

    metric_columns = metric_columns_for(args.dpi)
    samples_summary = summarise_by_sample(final_df, metric_columns)
    samples_path = os.path.join(args.output_dir, SAMPLES_SUMMARY_FILENAME)
    samples_summary.to_csv(samples_path, sep="\t", index=False)
    log.emit(f"Sample summary saved to {samples_path}")

    if args.save_images:
        log.emit(f"Prediction overlays saved in {predicted_dir}")

    images_processed = total_images - len(skipped_files)
    avg_per_image = total_elapsed / images_processed if images_processed else 0.0
    throughput = (images_processed / total_elapsed * 60) if total_elapsed > 0 else 0.0

    log.emit("")
    log.emit("Run summary")
    log.emit(f"  Images processed:   {images_processed}/{total_images}")
    log.emit(f"  Images skipped:     {len(skipped_files)}"
              + (f" ({', '.join(skipped_files)})" if skipped_files else ""))
    log.emit(f"  Total seeds kept:   {total_seeds_detected}")
    log.emit(f"  Total run time:     {format_duration(total_elapsed)}")
    log.emit(f"  Avg time/image:     {avg_per_image:.2f}s")
    log.emit(f"  Throughput:         {throughput:.1f} images/min")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate unfiltered seed metrics, per-accession summary stats, and optional overlays."
    )
    parser.add_argument("-i", "--input", required=True, help="Directory containing input images")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Root output directory for all generated tables and images")
    parser.add_argument("-w", "--weights",
                        default="data/barley/model_weights/mask_rcnn_barleyseeds_0040.h5",
                        help="Mask R-CNN weights path")
    parser.add_argument("--save-images", action="store_true",
                        help=f"Also save prediction overlays in <output-dir>/{PREDICTED_MASKS_DIRNAME}")
    parser.add_argument("--edge-crop", type=float, default=0.0,
                        help="Fraction removed from each left/right image edge (default: 0.0)")
    parser.add_argument("--min-score", type=float, default=0.95,
                        help="Minimum detection score (default: 0.95)")
    parser.add_argument("--min-contour-points", type=int, default=100,
                        help="Minimum contour point count (default: 100)")
    parser.add_argument("--alpha", type=float, default=0.35,
                        help="Mask overlay opacity from 0 to 1 (default: 0.35)")
    parser.add_argument("--max-instances", type=int, default=400,
                        help="Maximum final Mask R-CNN detections per image (default: 400)")
    parser.add_argument("--dpi", type=float, default=None,
                        help=(
                            "Scan resolution in dots per inch. When set, converts all "
                            "length/area columns to millimetres (mm_per_pixel = 25.4 / dpi) "
                            "and drops the pixel-based columns from the output. "
                            "Default: None (report pixels)."
                        ))
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        sys.exit(f"Error: input directory does not exist: {args.input}")
    if not os.path.isfile(args.weights):
        sys.exit(f"Error: weights file does not exist: {args.weights}")
    if not 0.0 <= args.edge_crop < 0.5:
        sys.exit("Error: --edge-crop must be at least 0 and less than 0.5.")
    if not 0.0 <= args.alpha <= 1.0:
        sys.exit("Error: --alpha must be between 0 and 1.")
    if args.max_instances < 1:
        sys.exit("Error: --max-instances must be at least 1.")
    if args.dpi is not None and args.dpi <= 0:
        sys.exit("Error: --dpi must be a positive number.")

    seed_path = os.path.join(args.output_dir, SEED_PARAMETERS_FILENAME)
    if os.path.exists(seed_path):
        response = input(f"Warning: {seed_path} already exists. Overwrite? (y/n): ")
        if response.lower() != "y":
            sys.exit("Process aborted by user.")

    log = RunLog()
    log.emit(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
    log.emit(f"Input directory:      {args.input}")
    log.emit(f"Output directory:     {args.output_dir}")
    log.emit(f"Weights:              {args.weights}")
    log.emit(f"Edge crop:            {args.edge_crop}")
    log.emit(f"Min score:            {args.min_score}")
    log.emit(f"Min contour points:   {args.min_contour_points}")
    log.emit(f"Max instances:        {args.max_instances}")
    log.emit(f"DPI (unit conversion):{'pixels (none)' if args.dpi is None else args.dpi}")
    log.emit(f"Save overlays:        {args.save_images}")
    log.emit("=" * 78)

    config = InferenceConfig()
    config.DETECTION_MAX_INSTANCES = args.max_instances
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
    log.emit(f"Loading weights from {args.weights}")
    model.load_weights(args.weights, by_name=True)

    process_images(args, model, log)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, RUN_LOG_FILENAME)
    log.write(log_path)
    print(f"Run log saved to {log_path}")


if __name__ == "__main__":
    main()
