"""Turn Mask R-CNN output into per-seed shape measurements."""

import cv2
import numpy as np
import pandas as pd

from .columns import COLOUR_COLUMNS, MEASURED_COLUMNS, ORIENTATION_FLAG_COLUMN, SHAPE_PROFILE_COLUMNS
from .shape import colour_lab, efd_coefficients, efd_column_names, shape_profile


def measure_instances(result, image_name, image_bgr, min_score, min_contour_points, efd_order=0,
                      base_end="blunt"):
    """Convert model output into one metric row and one overlay object per instance.

    All length/area/position values are computed in pixels; unit conversion
    happens later, once, on the assembled table. Colour is measured on
    image_bgr, which must not be contrast-enhanced. base_end ('blunt' or
    'narrow') says which end of each seed is its base. Returns the metrics
    DataFrame, the overlay instances, the raw (pre-filter) instance count
    reported by the model, and a DataFrame of EFD coefficients keyed by
    object_id (empty when efd_order is 0).
    """
    efd_columns = ["object_id"] + (efd_column_names(efd_order) if efd_order else [])
    masks = result.get("masks")
    scores = result.get("scores", [])
    raw_instance_count = 0 if masks is None or masks.ndim != 3 else masks.shape[-1]
    rows = []
    instances = []
    efd_rows = []

    if masks is None or masks.ndim != 3 or masks.shape[-1] == 0:
        return (pd.DataFrame(columns=MEASURED_COLUMNS), instances, raw_instance_count,
                pd.DataFrame(columns=efd_columns))

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

        profile, base_point = shape_profile(contour, base_end)
        colour = colour_lab(image_bgr, contour)
        if efd_order:
            efd_rows.append([object_id, *efd_coefficients(contour, efd_order)])

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
            *(profile[column] for column in SHAPE_PROFILE_COLUMNS),
            profile[ORIENTATION_FLAG_COLUMN],
            *(colour[column] for column in COLOUR_COLUMNS),
        ])
        instances.append({
            "object_id": object_id,
            "score": score,
            "mask": mask,
            "contour": contour,
            "base_point": base_point,
        })

    return (pd.DataFrame(rows, columns=MEASURED_COLUMNS), instances, raw_instance_count,
            pd.DataFrame(efd_rows, columns=efd_columns))
