"""Summary statistics tables."""


def summarise(df, group_columns, metric_columns, include_replicate_count=False):
    """Build a summary table with seed counts and per-metric mean/sd/min/max/median.

    Statistics use non-outlier seeds only; n_outliers counts the excluded ones.
    """
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

    groups = df.groupby(group_columns, dropna=False)
    n_outliers = groups["is_outlier"].sum().astype(int).rename("n_outliers")
    inliers = df[~df["is_outlier"].astype(bool)]
    table = (
        inliers.groupby(group_columns, dropna=False).agg(**aggregations)
        .reindex(n_outliers.index)  # keep groups whose seeds are all outliers
    )
    table["n_seeds"] = table["n_seeds"].fillna(0).astype(int)
    table.insert(1, "n_outliers", n_outliers)
    return table.reset_index()
