"""File-name parsing and spatial assignment of seeds to plant replicates.

Replicate assignment always works in pixel space, before any unit conversion.
"""

import os
import re

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


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


def top_to_bottom_order(labels, centroids):
    """Order cluster labels top to bottom by the mean y of their centroids."""
    unique_labels = np.unique(labels)
    mean_y = {label: centroids[labels == label, 1].mean() for label in unique_labels}
    return sorted(unique_labels, key=lambda label: mean_y[label])


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
    ordered_labels = top_to_bottom_order(labels, centroids)

    label_to_replicate = {
        label: replicate_id
        for label, replicate_id in zip(ordered_labels, replicate_ids)
    }
    return [label_to_replicate[label] for label in labels]
