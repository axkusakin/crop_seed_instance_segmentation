"""Spatial renumbering of seeds: left to right, top to bottom, from 1.

Mask R-CNN returns instances sorted by detection score, so its indices look
random on the image and change whenever a parameter changes. Seeds are
renumbered in reading order instead. As seeds lie scattered rather than on a
grid, they are grouped into horizontal rows first: sorted by centroid y, a
new row starts when a seed lies more than one median seed length below the
first seed of the current row. Within a row, seeds are numbered left to
right. Non-outliers are numbered first, then outliers continue the count in
the same reading order. The original Mask R-CNN index is kept as model_index.
"""

import numpy as np


def reading_order(centroids, row_height):
    """Return row indices into centroids, ordered left to right, top to bottom."""
    if len(centroids) == 0:
        return []
    by_y = np.argsort(centroids[:, 1], kind="stable")
    rows = []
    row_top = None
    for index in by_y:
        y = centroids[index, 1]
        if row_top is None or y - row_top > row_height:
            rows.append([])
            row_top = y
        rows[-1].append(index)
    return [index for row in rows for index in sorted(row, key=lambda i: centroids[i, 0])]


def renumber_seeds(metrics, instances):
    """Replace object_id with reading-order numbers, in metrics and instances.

    The previous object_id is kept in a model_index column. Returns the
    mapping {model_index: new object_id}.
    """
    if metrics.empty:
        metrics["model_index"] = metrics["object_id"]
        return {}

    centroids = metrics[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    is_outlier = metrics["is_outlier"].to_numpy(dtype=bool)
    inlier_lengths = metrics.loc[~is_outlier, "L_seed_length"]
    row_height = float(np.median(inlier_lengths if len(inlier_lengths) else metrics["L_seed_length"]))

    new_ids = np.zeros(len(metrics), dtype=int)
    next_id = 1
    for group in (np.flatnonzero(~is_outlier), np.flatnonzero(is_outlier)):
        for position in reading_order(centroids[group], row_height):
            new_ids[group[position]] = next_id
            next_id += 1

    old_ids = metrics["object_id"].to_numpy()
    mapping = {int(old): int(new) for old, new in zip(old_ids, new_ids)}
    metrics["model_index"] = old_ids
    metrics["object_id"] = new_ids
    for instance in instances:
        instance["object_id"] = mapping[instance["object_id"]]
    return mapping
