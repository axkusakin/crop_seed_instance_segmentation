#!/usr/bin/env python3
"""Detect barley grains, save per-seed metrics with replicate IDs, summary
statistics, and optionally save mask overlays -- all under one output folder.

One Mask R-CNN inference pass per image. No IQR filtering is applied -- every
predicted object that passes --min-score and --min-contour-points is written
to the per-seed table.

File-name parsing
------------------
    SV1_18-19_015_1-2-3.jpg  -> sample_id "SV1_18-19_015", replicate IDs [1, 2, 3]
    SV2_19-20_043_1-9-12.jpg -> sample_id "SV2_19-20_043", replicate IDs [1, 9, 12]
    SV_022_5-4-8.jpg         -> sample_id "SV_022",        replicate IDs [5, 4, 8]
    SV_007_4.jpg              -> sample_id "SV_007",        replicate ID  [4] (single plant)

sample_id  = every underscore-separated part of the file name except the
             trailing replicate token.
replicate_id = one of the dash-separated numbers in that trailing token.

Replicate assignment
---------------------
Each image may contain 1-3 plant replicates spread with an empty gap between
them. For images with more than one replicate ID, seed centroids (always
computed and clustered in pixel space, regardless of --dpi) are connected
into a minimum spanning tree (MST); removing the (k - 1) longest edges from
any spanning tree always yields exactly k connected components, so this
deterministically splits the seeds into k spatial groups regardless of
whether the gap runs horizontally, vertically, or diagonally. Groups are
then ordered along the principal spread axis of the centroids (via PCA) and
matched, in that order, to the replicate IDs as listed in the file name --
the sequence in which the plants were physically laid on the imaging bed.

Units
-----
By default, all length/area/position measurements are in pixels. Pass --dpi
to convert them to millimetres, assuming every input image was scanned at
that fixed resolution (mm_per_pixel = 25.4 / dpi). When --dpi is given, the
final tables report ONLY millimetre-based columns (suffixed _mm or _mm2)
plus the dimensionless ratios (LWR, eccentricity, solidity, circularity) --
pixel columns are dropped from the output. Conversion happens once, on the
fully assembled table, after replicate assignment and visualisation (which
always operate in pixel space).

Output structure
-----------------
Everything is written under a single --output-dir:

    <output-dir>/
        seed_parameters.tsv      one row per detected seed
        replicates_summary.tsv   summary stats per (sample_id, replicate_id)
        samples_summary.tsv      summary stats per sample_id (all replicates combined)
        run_log.txt              run parameters and a per-run summary
        predicted_masks/         only created if --save-images is passed

Examples
--------
# Tables only, pixels
python grain_metrics_and_visualize_with_replicates.py -i images -o output_dir

# Tables plus matched prediction overlays, millimetres (500 DPI scan)
python grain_metrics_and_visualize_with_replicates.py -i images -o output_dir \
    --save-images --dpi 500

# Exclude 8% from both lateral edges, allow up to 400 detections per image
python grain_metrics_and_visualize_with_replicates.py -i images -o output_dir \
    --save-images --edge-crop 0.08 --max-instances 400 --dpi 500
"""

import argparse
import os
import re
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

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

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


# Columns produced directly by detection/measurement (sample_id/replicate_id added later).
# Always in pixels at this stage.
METRIC_COLUMNS = [
    "file_name", "object_id", "detection_score", "AS_seed_area",
    "L_seed_length", "W_seed_width", "LWR_length_to_width_ratio",
    "eccentricity", "solidity", "PL_perimeter_length", "CS_seed_circularity",
    "centroid_x", "centroid_y",
]

# Final per-seed column order (pixel-named; renamed by convert_to_millimetres if --dpi is set).
FINAL_COLUMNS = [
    "file_name", "sample_id", "object_id", "detection_score", "replicate_id",
    "AS_seed_area", "L_seed_length", "W_seed_width", "LWR_length_to_width_ratio",
    "eccentricity", "solidity", "PL_perimeter_length", "CS_seed_circularity",
    "centroid_x", "centroid_y",
]

# Pixel-valued columns and how they scale when converted to millimetres:
# length-like and position columns scale linearly (^1); area columns scale by area (^2).
LENGTH_COLUMNS = ["L_seed_length", "W_seed_width", "PL_perimeter_length"]
AREA_COLUMNS = ["AS_seed_area"]
POSITION_COLUMNS = ["centroid_x", "centroid_y"]

# Dimensionless columns: never converted, never renamed.
RATIO_COLUMNS = [
    "LWR_length_to_width_ratio", "eccentricity", "solidity", "CS_seed_circularity",
]

# Shape metrics reported in the summary tables (position columns are excluded).
SHAPE_METRIC_COLUMNS = LENGTH_COLUMNS + AREA_COLUMNS + RATIO_COLUMNS

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PREDICTED_MASKS_DIRNAME = "predicted_masks"
SEED_PARAMETERS_FILENAME = "seed_parameters.tsv"
REPLICATES_SUMMARY_FILENAME = "replicates_summary.tsv"
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
# Detection and measurement
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

    All length/area/position values are computed in pixels; unit conversion
    happens later, once, on the assembled table. Returns the metrics
    DataFrame, the overlay instances, and the raw (pre-filter) instance
    count reported by the model, for console diagnostics.
    """
    masks = result.get("masks")
    scores = result.get("scores", [])
    raw_instance_count = 0 if masks is None or masks.ndim != 3 else masks.shape[-1]
    rows = []
    instances = []

    if masks is None or masks.ndim != 3 or masks.shape[-1] == 0:
        return pd.DataFrame(columns=METRIC_COLUMNS), instances, raw_instance_count

    for object_id in range(masks.shape[-1]):
        score = float(scores[object_id])
        if score < min_score:
            continue

        mask = masks[:, :, object_id].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if len(contour) >= min_contour_points]
        if not contours:
            continue

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

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]

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
            centroid_x,
            centroid_y,
        ])
        instances.append({
            "object_id": object_id,
            "score": score,
            "mask": mask,
            "contour": contour,
        })

    return pd.DataFrame(rows, columns=METRIC_COLUMNS), instances, raw_instance_count


# --------------------------------------------------------------------------
# File-name parsing: sample_id + replicate IDs
# --------------------------------------------------------------------------

def parse_sample_and_replicates(file_name):
    """Split a file name into (sample_id, ordered replicate_ids)."""
    base = os.path.splitext(file_name)[0]
    sample_id, _, last_token = base.rpartition("_")
    if not sample_id:
        raise ValueError(f"file name has no sample/replicate separator: {file_name!r}")
    if not re.fullmatch(r"\d+(-\d+)*", last_token):
        raise ValueError(
            f"could not parse replicate token from file name (trailing token was {last_token!r})"
        )
    replicate_ids = [int(part) for part in last_token.split("-")]
    return sample_id, replicate_ids


# --------------------------------------------------------------------------
# Replicate assignment (spatial gap clustering, always in pixel space)
# --------------------------------------------------------------------------

def principal_axis_order(labels, centroids):
    """Order cluster labels along the first principal component of centroids."""
    centered = centroids - centroids.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    projections = centered @ principal_axis

    unique_labels = np.unique(labels)
    mean_projection = {
        label: projections[labels == label].mean() for label in unique_labels
    }
    return sorted(unique_labels, key=lambda label: mean_projection[label])


def split_into_groups(centroids, num_groups):
    """Split centroids into num_groups clusters using an MST gap cut."""
    num_points = centroids.shape[0]
    if num_points < num_groups:
        raise ValueError(
            f"only {num_points} seed(s) detected but {num_groups} replicate IDs expected"
        )

    distances = squareform(pdist(centroids))
    mst = minimum_spanning_tree(csr_matrix(distances))
    mst_coo = mst.tocoo()

    edges = list(zip(mst_coo.row, mst_coo.col, mst_coo.data))
    edges.sort(key=lambda edge: edge[2], reverse=True)

    edges_to_cut = edges[: num_groups - 1]
    cut_set = {(row, col) for row, col, _ in edges_to_cut} | {
        (col, row) for row, col, _ in edges_to_cut
    }

    kept_row, kept_col, kept_data = [], [], []
    for row, col, weight in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        if (row, col) in cut_set:
            continue
        kept_row.append(row)
        kept_col.append(col)
        kept_data.append(weight)

    pruned = csr_matrix(
        (kept_data, (kept_row, kept_col)), shape=(num_points, num_points)
    )
    n_components, labels = connected_components(pruned, directed=False)

    if n_components != num_groups:
        raise RuntimeError(
            f"expected {num_groups} groups but obtained {n_components} "
            "(this should not happen for a valid spanning tree cut)"
        )
    return labels


def assign_replicates(metrics_df, replicate_ids):
    """Return a list of replicate IDs aligned to metrics_df's row order."""
    num_replicates = len(replicate_ids)
    if num_replicates == 1:
        return [replicate_ids[0]] * len(metrics_df)

    centroids = metrics_df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    labels = split_into_groups(centroids, num_replicates)
    ordered_labels = principal_axis_order(labels, centroids)

    label_to_replicate = {
        label: replicate_id
        for label, replicate_id in zip(ordered_labels, replicate_ids)
    }
    return [label_to_replicate[label] for label in labels]


# --------------------------------------------------------------------------
# Visualisation (always drawn in pixel space, before any conversion)
# --------------------------------------------------------------------------

def draw_predictions(image_bgr, instances, replicate_by_object_id, alpha):
    """Draw every retained object, labelled with its object_id and replicate_id."""
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

        replicate_id = replicate_by_object_id.get(object_id, "?")
        label = f"{object_id} (rep {replicate_id}): {instance['score']:.2f}"
        cv2.putText(output, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)

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
    """Convert pixel-based columns to millimetres, replacing the pixel columns.

    Length-like and position columns scale by mm_per_pixel; area columns
    scale by mm_per_pixel squared. Ratio columns are left untouched. Returns
    a new DataFrame with pixel columns replaced by their _mm / _mm2
    equivalents.
    """
    mm_per_pixel = 25.4 / dpi
    converted = df.copy()

    rename_map = {}
    for column in LENGTH_COLUMNS + POSITION_COLUMNS:
        converted[column] = converted[column] * mm_per_pixel
        rename_map[column] = f"{column}_mm"
    for column in AREA_COLUMNS:
        converted[column] = converted[column] * (mm_per_pixel ** 2)
        rename_map[column] = f"{column}_mm2"

    return converted.rename(columns=rename_map)


def shape_metric_columns_for(dpi):
    """Return the shape-metric column names to summarise, given the active unit."""
    if dpi is None:
        return SHAPE_METRIC_COLUMNS
    length_mm = [f"{column}_mm" for column in LENGTH_COLUMNS]
    area_mm2 = [f"{column}_mm2" for column in AREA_COLUMNS]
    return length_mm + area_mm2 + RATIO_COLUMNS


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

def summarise(df, group_columns, metric_columns, include_replicate_count):
    """Build a summary table with seed counts and per-metric mean/sd/min/max/median."""
    aggregations = {"n_seeds": ("object_id", "count")}
    if include_replicate_count:
        aggregations["n_replicates"] = ("replicate_id", "nunique")
    aggregations["mean_detection_score"] = ("detection_score", "mean")

    for metric in metric_columns:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_sd"] = (metric, "std")
        aggregations[f"{metric}_min"] = (metric, "min")
        aggregations[f"{metric}_max"] = (metric, "max")
        aggregations[f"{metric}_median"] = (metric, "median")

    return df.groupby(group_columns, dropna=False).agg(**aggregations).reset_index()


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
    parsing_failures = []
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

        replicate_by_object_id = {}
        replicate_text = "sample/replicate: n/a"
        if not metrics.empty:
            try:
                sample_id, replicate_ids = parse_sample_and_replicates(filename)
                metrics["sample_id"] = sample_id
                metrics["replicate_id"] = assign_replicates(metrics, replicate_ids)
                replicate_by_object_id = dict(
                    zip(metrics["object_id"], metrics["replicate_id"])
                )
                replicate_text = f"sample {sample_id!r}, {len(replicate_ids)} replicate(s)"
            except (ValueError, RuntimeError) as error:
                log.emit(f"Warning: {filename}: {error}. Leaving sample_id/replicate_id blank.")
                metrics["sample_id"] = pd.NA
                metrics["replicate_id"] = pd.NA
                parsing_failures.append(filename)
                replicate_text = "replicate parsing FAILED"
            tables.append(metrics)

        if args.save_images:
            visualisation = draw_predictions(
                image_bgr, instances, replicate_by_object_id, args.alpha
            )
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
            f"{width}x{height}px | raw {raw_instance_count} | retained {len(instances)} | "
            f"{replicate_text} | {score_text} | {elapsed:.2f}s | ETA {eta_text}"
        )

    total_elapsed = time.perf_counter() - run_start
    log.emit("-" * 78)

    if not tables:
        sys.exit("Error: no objects passed the selected score and contour filters. Nothing to save.")

    os.makedirs(args.output_dir, exist_ok=True)

    final_df = pd.concat(tables, ignore_index=True)
    final_df = final_df.sort_values(
        by=["file_name", "replicate_id", "object_id"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    final_df = final_df[FINAL_COLUMNS]

    if args.dpi is not None:
        final_df = convert_to_millimetres(final_df, args.dpi)
        log.emit(f"Converted pixel measurements to millimetres using {args.dpi} DPI.")

    seed_path = os.path.join(args.output_dir, SEED_PARAMETERS_FILENAME)
    final_df.to_csv(seed_path, sep="\t", index=False)
    log.emit(f"Per-seed table saved to {seed_path}")

    labelled_df = final_df.dropna(subset=["sample_id", "replicate_id"])
    replicates_created = False
    if labelled_df.empty:
        log.emit("Warning: no rows had a valid sample_id/replicate_id; summary tables were not created.")
    else:
        metric_columns = shape_metric_columns_for(args.dpi)

        replicates_summary = summarise(
            labelled_df, group_columns=["sample_id", "replicate_id"],
            metric_columns=metric_columns, include_replicate_count=False,
        )
        replicates_path = os.path.join(args.output_dir, REPLICATES_SUMMARY_FILENAME)
        replicates_summary.to_csv(replicates_path, sep="\t", index=False)
        log.emit(f"Replicate summary saved to {replicates_path}")

        samples_summary = summarise(
            labelled_df, group_columns=["sample_id"],
            metric_columns=metric_columns, include_replicate_count=True,
        )
        samples_path = os.path.join(args.output_dir, SAMPLES_SUMMARY_FILENAME)
        samples_summary.to_csv(samples_path, sep="\t", index=False)
        log.emit(f"Sample summary saved to {samples_path}")
        replicates_created = True

    if args.save_images:
        log.emit(f"Prediction overlays saved in {predicted_dir}")

    images_processed = total_images - len(skipped_files)
    avg_per_image = total_elapsed / images_processed if images_processed else 0.0
    throughput = (images_processed / total_elapsed * 60) if total_elapsed > 0 else 0.0
    unique_samples = final_df["sample_id"].nunique(dropna=True)
    unique_replicates = (
        final_df.dropna(subset=["sample_id", "replicate_id"])
        .drop_duplicates(subset=["sample_id", "replicate_id"])
        .shape[0]
    )

    log.emit("")
    log.emit("Run summary")
    log.emit(f"  Images processed:      {images_processed}/{total_images}")
    log.emit(f"  Images skipped (I/O):  {len(skipped_files)}"
              + (f" ({', '.join(skipped_files)})" if skipped_files else ""))
    log.emit(f"  Replicate parse fails: {len(parsing_failures)}"
              + (f" ({', '.join(parsing_failures)})" if parsing_failures else ""))
    log.emit(f"  Total seeds kept:      {total_seeds_detected}")
    log.emit(f"  Distinct samples:      {unique_samples}")
    log.emit(f"  Distinct replicates:   {unique_replicates}")
    log.emit(f"  Summary tables built:  {replicates_created}")
    log.emit(f"  Total run time:        {format_duration(total_elapsed)}")
    log.emit(f"  Avg time/image:        {avg_per_image:.2f}s")
    log.emit(f"  Throughput:            {throughput:.1f} images/min")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect barley grains, save per-seed metrics with replicate IDs, build "
            "replicate- and sample-level summary tables, and optionally save overlays."
        )
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
                            "length/area/position columns to millimetres "
                            "(mm_per_pixel = 25.4 / dpi) and drops the pixel-based columns "
                            "from the output. Default: None (report pixels)."
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
