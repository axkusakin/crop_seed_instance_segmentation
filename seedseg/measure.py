"""Turn Mask R-CNN output into per-seed shape measurements."""

import cv2
import numpy as np
import pandas as pd

from .columns import MEASURED_COLUMNS


def measure_instances(result, image_name, min_score, min_contour_points):
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
        return pd.DataFrame(columns=MEASURED_COLUMNS), instances, raw_instance_count

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

    return pd.DataFrame(rows, columns=MEASURED_COLUMNS), instances, raw_instance_count
