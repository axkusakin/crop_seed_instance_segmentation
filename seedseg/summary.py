"""Summary statistics tables."""

from .columns import SUMMARY_STATISTICS

_PANDAS_STATISTIC = {"mean": "mean", "sd": "std", "median": "median", "min": "min", "max": "max"}


def summarise(df, group_columns, metric_columns, include_replicate_count=False):
    """Build a summary table with seed counts and per-metric mean/sd/median/min/max.

    Statistics use non-outlier seeds only; n_outliers counts the excluded ones.
    """
    aggregations = {"n_seeds": ("object_id", "count")}
    if include_replicate_count:
        aggregations["n_replicates"] = ("replicate_id", "nunique")
    aggregations["mean_detection_score"] = ("detection_score", "mean")

    # Grouped by metric, in the fixed order given by metric_columns and SUMMARY_STATISTICS.
    for metric in metric_columns:
        for statistic in SUMMARY_STATISTICS:
            aggregations[f"{metric}_{statistic}"] = (metric, _PANDAS_STATISTIC[statistic])

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
