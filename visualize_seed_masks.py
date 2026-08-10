#!/usr/bin/env python3
"""Visualize Mask R-CNN barley-seed predictions as mask overlays and contours.

Each output image contains a transparent instance-mask overlay, a colored contour,
and an instance label (ID: confidence).  By default, this uses the same score and
contour-size filters as grain_metrics.py, so displayed seeds correspond to the
ones retained for metric extraction.
"""

import argparse
import os
import sys

import cv2
import numpy as np

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
    DETECTION_MAX_INSTANCES = 400
    RPN_NMS_THRESHOLD = 0.4


CONFIG = InferenceConfig()
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def instance_colour(index):
    """Return a reproducible high-contrast BGR colour for an instance."""
    hue = (index * 47) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(x) for x in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_predictions(image_bgr, result, min_score, min_contour_points, alpha):
    """Draw accepted predicted masks, external contours, and labels."""
    output = image_bgr.copy()
    masks = result.get("masks")
    scores = result.get("scores", [])

    if masks is None or masks.ndim != 3 or masks.shape[-1] == 0:
        return output, 0

    accepted = []
    for prediction_id in range(masks.shape[-1]):
        score = float(scores[prediction_id])
        if score < min_score:
            continue

        mask = masks[:, :, prediction_id].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if len(c) >= min_contour_points]
        if contours:
            accepted.append((prediction_id, score, mask, contours))

    overlay = output.copy()
    for display_id, (_, _, mask, _) in enumerate(accepted, start=1):
        overlay[mask.astype(bool)] = instance_colour(display_id)
    output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)

    for display_id, (prediction_id, score, _, contours) in enumerate(accepted, start=1):
        colour = instance_colour(display_id)
        cv2.drawContours(output, contours, -1, colour, 2, lineType=cv2.LINE_AA)
        largest = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest)
        if moments["m00"]:
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
        else:
            x, y = largest.reshape(-1, 2)[0]
        label = f"{display_id}: {score:.2f}"
        cv2.putText(output, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(output, f"Seeds retained: {len(accepted)}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, f"Seeds retained: {len(accepted)}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
    return output, len(accepted)


def process_directory(input_dir, output_dir, model, min_score, min_contour_points, alpha, edge_crop):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [
        name for name in sorted(os.listdir(input_dir))
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        sys.exit(f"Error: no supported image files found in {input_dir}")

    total = 0
    for name in image_files:
        image_path = os.path.join(input_dir, name)
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"Warning: could not read {image_path}; skipping.")
            continue
        
        # Remove the specified fraction from both lateral image edges.
        height, width = image_bgr.shape[:2]
        edge_px = int(round(width * edge_crop))
        if edge_px > 0:
            image_bgr = image_bgr[:, edge_px:width - edge_px]

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = model.detect([image_rgb], verbose=0)[0]
        visualisation, count = draw_predictions(
            image_bgr, result, min_score, min_contour_points, alpha
        )

        output_name = f"{os.path.splitext(name)[0]}_predicted.png"
        output_path = os.path.join(output_dir, output_name)
        if not cv2.imwrite(output_path, visualisation):
            print(f"Warning: could not write {output_path}")
            continue
        print(f"{name}: saved {output_path} ({count} retained seeds)")
        total += count

    print(f"Done. Saved visualisations in {output_dir}; total retained seeds: {total}")


def main():
    parser = argparse.ArgumentParser(
        description="Save Mask R-CNN seed-mask overlays and contours for every input image."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Directory containing input seed images")
    parser.add_argument("-w", "--weights",
                        default="data/barley/model_weights/mask_rcnn_barleyseeds_0040.h5",
                        help="Path to Mask R-CNN weights")
    parser.add_argument("-d", "--output-dir", default="predicted_images",
                        help="Directory for visualised PNG images (default: predicted_images)")
    parser.add_argument("--min-score", type=float, default=0.95,
                        help="Minimum detection score (default: 0.95, matching grain_metrics.py)")
    parser.add_argument("--min-contour-points", type=int, default=100,
                        help="Minimum contour point count (default: 100, matching grain_metrics.py)")
    parser.add_argument("--alpha", type=float, default=0.35,
                        help="Mask overlay opacity from 0 to 1 (default: 0.35)")
    parser.add_argument("--edge-crop", type=float, default=0.0,
                        help="Fraction of image width removed from each left/right edge (default: 0.0)")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        sys.exit(f"Error: input directory does not exist: {args.input}")
    if not os.path.isfile(args.weights):
        sys.exit(f"Error: weights file does not exist: {args.weights}")
    if not 0.0 <= args.alpha <= 1.0:
        sys.exit("Error: --alpha must be between 0 and 1.")
    if not 0.0 <= args.edge_crop < 0.5:
        sys.exit("Error: --edge-crop must be at least 0 and less than 0.5.")

    model = modellib.MaskRCNN(mode="inference", config=CONFIG, model_dir="")
    print(f"Loading weights from {args.weights}")
    model.load_weights(args.weights, by_name=True)
    process_directory(args.input, args.output_dir, model, args.min_score,
                      args.min_contour_points, args.alpha, args.edge_crop)


if __name__ == "__main__":
    main()
