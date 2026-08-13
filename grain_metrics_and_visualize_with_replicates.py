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
them. For images with more than one replicate ID, seed centroids are
connected into a minimum spanning tree (MST); removing the (k - 1) longest
edges from any spanning tree always yields exactly k connected components,
so this deterministically splits the seeds into k spatial groups regardless
of whether the gap runs horizontally, vertically, or diagonally. Groups are
then ordered along the principal spread axis of the centroids (via PCA) and
matched, in that order, to the replicate IDs as listed in the file name --
the sequence in which the plants were physically laid on the imaging bed.

Output structure
-----------------
Everything is written under a single --output-dir:

    <output-dir>/
        seed_parameters.tsv      one row per detected seed
        replicates_summary.tsv   summary stats per (sample_id, replicate_id)
        samples_summary.tsv      summary stats per sample_id (all replicates combined)
        predicted_masks/         only created if --save-images is passed

Examples
--------
# Tables only
python grain_metrics_and_visualize_with_replicates.py -i images -o output_dir

# Tables plus matched prediction overlays
python grain_metrics_and_visualize_with_replicates.py -i images -o output_dir --save-images

# Exclude 8% from both lateral edges, allow up to 400 detections per image
python grain_metrics_and_visualize_with_replicates.py -i images -o output_dir \
    --save-images --edge-crop 0.08 --max-instances 400
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa: F401

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
METRIC_COLUMNS = [
    "file_name", "object_id", "detection_score", "AS_seed_area",
    "L_seed_length", "W_seed_width", "LWR_length_to_width_ratio",
    "eccentricity", "solidity", "PL_perimeter_length", "CS_seed_circularity",
    "centroid_x", "centroid_y",
]

# Final per-seed column order.
FINAL_COLUMNS = [
    "file_name", "sample_id", "object_id", "detection_score", "replicate_id",
    "AS_seed_area", "L_seed_length", "W_seed_width", "LWR_length_to_width_ratio",
    "eccentricity", "solidity", "PL_perimeter_length", "CS_seed_circularity",
    "centroid_x", "centroid_y",
]

# Metrics summarised in replicates_summary.tsv and samples_summary.tsv.
SUMMARY_METRICS = [
    "AS_seed_area", "L_seed_length", "W_seed_width", "LWR_length_to_width_ratio",
    "eccentricity", "solidity", "PL_perimeter_length", "CS_seed_circularity",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PREDICTED_MASKS_DIRNAME = "predicted_masks"
SEED_PARAMETERS_FILENAME = "seed_parameters.tsv"
REPLICATES_SUMMARY_FILENAME = "replicates_summary.tsv"
SAMPLES_SUMMARY_FILENAME = "samples_summary.tsv"


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
    """Convert model output into one metric row and one overlay object per instance."""
    masks = result.get("masks")
    scores = result.get("scores", [])
    rows = []
    instances = []

    if masks is None or masks.ndim != 3 or masks.shape[-1] == 0:
        return pd.DataFrame(columns=METRIC_COLUMNS), instances

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

    return pd.DataFrame(rows, columns=METRIC_COLUMNS), instances


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
# Replicate assignment (spatial gap clustering)
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
# Visualisation
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
# Summary statistics
# --------------------------------------------------------------------------

def summarise(df, group_columns, include_replicate_count):
    """Build a summary table with seed counts and per-metric mean/sd/min/max/median."""
    aggregations = {"n_seeds": ("object_id", "count")}
    if include_replicate_count:
        aggregations["n_replicates"] = ("replicate_id", "nunique")
    aggregations["mean_detection_score"] = ("detection_score", "mean")

    for metric in SUMMARY_METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_sd"] = (metric, "std")
        aggregations[f"{metric}_min"] = (metric, "min")
        aggregations[f"{metric}_max"] = (metric, "max")
        aggregations[f"{metric}_median"] = (metric, "median")

    summary = df.groupby(group_columns, dropna=False).agg(**aggregations).reset_index()
    return summary


# --------------------------------------------------------------------------
# Main processing loop
# --------------------------------------------------------------------------

def process_images(args, model):
    predicted_dir = os.path.join(args.output_dir, PREDICTED_MASKS_DIRNAME)
    if args.save_images:
        os.makedirs(predicted_dir, exist_ok=True)

    image_files = [
        filename for filename in sorted(os.listdir(args.input))
        if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        sys.exit(f"Error: no supported images found in {args.input}")

    tables = []
    for filename in image_files:
        image_path = os.path.join(args.input, filename)
        original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original_bgr is None:
            print(f"Warning: could not read {image_path}; skipping.")
            continue

        image_bgr = crop_lateral_edges(original_bgr, args.edge_crop)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = model.detect([image_rgb], verbose=0)[0]

        metrics, instances = detect_and_measure(
            result, filename, args.min_score, args.min_contour_points
        )

        replicate_by_object_id = {}
        if not metrics.empty:
            try:
                sample_id, replicate_ids = parse_sample_and_replicates(filename)
                metrics["sample_id"] = sample_id
                metrics["replicate_id"] = assign_replicates(metrics, replicate_ids)
                replicate_by_object_id = dict(
                    zip(metrics["object_id"], metrics["replicate_id"])
                )
            except (ValueError, RuntimeError) as error:
                print(f"Warning: {filename}: {error}. Leaving sample_id/replicate_id blank.")
                metrics["sample_id"] = pd.NA
                metrics["replicate_id"] = pd.NA
            tables.append(metrics)

        if args.save_images:
            visualisation = draw_predictions(
                image_bgr, instances, replicate_by_object_id, args.alpha
            )
            output_name = f"{os.path.splitext(filename)[0]}_predicted.png"
            output_path = os.path.join(predicted_dir, output_name)
            if not cv2.imwrite(output_path, visualisation):
                print(f"Warning: could not write {output_path}")

        print(f"{filename}: {len(instances)} objects detected")

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

    seed_path = os.path.join(args.output_dir, SEED_PARAMETERS_FILENAME)
    final_df.to_csv(seed_path, sep="\t", index=False)
    print(f"Per-seed table saved to {seed_path}")

    labelled_df = final_df.dropna(subset=["sample_id", "replicate_id"])
    if labelled_df.empty:
        print("Warning: no rows had a valid sample_id/replicate_id; summary tables were not created.")
        return

    replicates_summary = summarise(
        labelled_df, group_columns=["sample_id", "replicate_id"], include_replicate_count=False
    )
    replicates_path = os.path.join(args.output_dir, REPLICATES_SUMMARY_FILENAME)
    replicates_summary.to_csv(replicates_path, sep="\t", index=False)
    print(f"Replicate summary saved to {replicates_path}")

    samples_summary = summarise(
        labelled_df, group_columns=["sample_id"], include_replicate_count=True
    )
    samples_path = os.path.join(args.output_dir, SAMPLES_SUMMARY_FILENAME)
    samples_summary.to_csv(samples_path, sep="\t", index=False)
    print(f"Sample summary saved to {samples_path}")

    if args.save_images:
        print(f"Prediction overlays saved in {predicted_dir}")


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

    seed_path = os.path.join(args.output_dir, SEED_PARAMETERS_FILENAME)
    if os.path.exists(seed_path):
        response = input(f"Warning: {seed_path} already exists. Overwrite? (y/n): ")
        if response.lower() != "y":
            sys.exit("Process aborted by user.")

    config = InferenceConfig()
    config.DETECTION_MAX_INSTANCES = args.max_instances
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
    print(f"Loading weights from {args.weights}")
    model.load_weights(args.weights, by_name=True)
    process_images(args, model)


if __name__ == "__main__":
    main()
