#!/usr/bin/env python3
"""Calculate barley-grain metrics and optionally save matched mask overlays.

No IQR filtering is applied. Every predicted object that passes --min-score and
--min-contour-points is written to the TSV and, when --save-images is used,
drawn in the corresponding overlay. The image label is the exact zero-based
object_id in the TSV, enabling later manual removal of selected objects.

Examples
--------
# TSV only
python grain_metrics_no_iqr.py -i images -o results.tsv

# TSV plus one matched prediction overlay per image
python grain_metrics_no_iqr.py -i images -o results.tsv --save-images

# Exclude 8% from both lateral edges and allow up to 400 final detections
python grain_metrics_no_iqr.py -i images -o results.tsv --save-images \
    --edge-crop 0.08 --max-instances 400
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa: F401

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


COLUMNS = [
    "file_name", "object_id", "detection_score", "AS_seed_area",
    "L_seed_length", "W_seed_width", "LWR_length_to_width_ratio",
    "eccentricity", "solidity", "PL_perimeter_length", "CS_seed_circularity",
]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


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
        return pd.DataFrame(columns=COLUMNS), instances

    for object_id in range(masks.shape[-1]):
        score = float(scores[object_id])
        if score < min_score:
            continue

        mask = masks[:, :, object_id].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if len(contour) >= min_contour_points]
        if not contours:
            continue

        # One model instance yields one TSV row. Use the largest connected
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

    return pd.DataFrame(rows, columns=COLUMNS), instances


def draw_predictions(image_bgr, instances, alpha):
    """Draw every TSV object, labelled with its exact TSV object_id."""
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

    summary = f"Objects in TSV: {len(instances)}"
    cv2.putText(output, summary, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, summary, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def process_images(args, model):
    if args.save_images:
        os.makedirs(args.predicted_dir, exist_ok=True)

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
        if not metrics.empty:
            tables.append(metrics)

        if args.save_images:
            visualisation = draw_predictions(image_bgr, instances, args.alpha)
            output_name = f"{os.path.splitext(filename)[0]}_predicted.png"
            output_path = os.path.join(args.predicted_dir, output_name)
            if not cv2.imwrite(output_path, visualisation):
                print(f"Warning: could not write {output_path}")

        print(f"{filename}: {len(instances)} objects written to TSV")

    if not tables:
        sys.exit("Error: no objects passed the selected score and contour filters. Nothing to save.")

    final_df = pd.concat(tables, ignore_index=True)
    final_df.to_csv(args.output, sep="\t", index=False)
    print(f"Results saved to {args.output}")
    if args.save_images:
        print(f"Matched prediction visualisations saved in {args.predicted_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate unfiltered seed metrics and optional overlays with TSV-matched object IDs."
    )
    parser.add_argument("-i", "--input", required=True, help="Directory containing input images")
    parser.add_argument("-o", "--output", required=True, help="Output TSV path")
    parser.add_argument("-w", "--weights",
                        default="data/barley/model_weights/mask_rcnn_barleyseeds_0040.h5",
                        help="Mask R-CNN weights path")
    parser.add_argument("--save-images", action="store_true",
                        help="Save prediction overlays in addition to the TSV")
    parser.add_argument("-d", "--predicted-dir", default="predicted_images",
                        help="Overlay output directory (default: predicted_images)")
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
    if os.path.exists(args.output):
        response = input(f"Warning: {args.output} already exists. Overwrite? (y/n): ")
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
