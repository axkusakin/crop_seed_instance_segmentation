"""Summary statistics tables."""


def summarise(df, group_columns, metric_columns, include_replicate_count=False):
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
