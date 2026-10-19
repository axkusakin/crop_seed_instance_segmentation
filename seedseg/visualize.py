"""Prediction overlays, always drawn in pixel space."""

import cv2
import numpy as np

OUTLIER_COLOUR = (0, 0, 255)  # BGR red


def instance_colour(object_id):
    """Return a reproducible high-contrast BGR colour for an object ID."""
    hue = ((object_id + 1) * 47) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_predictions(image_bgr, instances, alpha, replicate_by_object_id=None, outlier_ids=()):
    """Draw every retained object, labelled with its object_id.

    When replicate_by_object_id is given, labels also show each object's
    replicate_id ("?" for objects without one). Objects in outlier_ids are
    drawn as an unfilled red contour with an "OUTLIER" label.
    """
    output = image_bgr.copy()
    overlay = output.copy()

    for instance in instances:
        if instance["object_id"] not in outlier_ids:
            overlay[instance["mask"].astype(bool)] = instance_colour(instance["object_id"])
    output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)

    font_scale = 0.45 if replicate_by_object_id is None else 0.42
    for instance in instances:
        object_id = instance["object_id"]
        contour = instance["contour"]
        is_outlier = object_id in outlier_ids
        colour = OUTLIER_COLOUR if is_outlier else instance_colour(object_id)
        cv2.drawContours(output, [contour], -1, colour, 2, cv2.LINE_AA)

        moments = cv2.moments(contour)
        if moments["m00"]:
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
        else:
            x, y = contour.reshape(-1, 2)[0]

        if replicate_by_object_id is None:
            label = f"{object_id}: {instance['score']:.2f}"
        else:
            replicate_id = replicate_by_object_id.get(object_id, "?")
            label = f"{object_id} (rep {replicate_id}): {instance['score']:.2f}"
        if is_outlier:
            label += " OUTLIER"
        cv2.putText(output, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    summary = f"Objects in table: {len(instances)}"
    if outlier_ids:
        summary += f" (outliers: {len(outlier_ids)})"
    cv2.putText(output, summary, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, summary, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return output
