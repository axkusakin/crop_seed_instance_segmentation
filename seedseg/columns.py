"""Column names, their fixed order, and output file names shared across the pipeline.

The order of every output table is defined here and nowhere else: new
metrics are added to the lists below, so existing columns never move
unexpectedly. Summary tables follow METRIC_ORDER, grouped by metric
(<metric>_mean, _sd, _median, _min, _max).
"""

# Size: in pixels until converted (length-like ^1, area ^2) when --dpi is set.
LENGTH_COLUMNS = ["L_seed_length", "W_seed_width", "PL_perimeter_length"]
AREA_COLUMNS = ["AS_seed_area"]
POSITION_COLUMNS = ["centroid_x", "centroid_y"]

# Dimensionless shape ratios: never converted, never renamed.
RATIO_COLUMNS = [
    "LWR_length_to_width_ratio", "eccentricity", "solidity", "CS_seed_circularity",
]

# Width profile and asymmetry, dimensionless (see seedseg/shape.py).
SHAPE_PROFILE_COLUMNS = [
    "lateral_area_asymmetry", "lateral_mirror_iou", "longitudinal_area_asymmetry",
    "width_profile_skewness", "max_width_position", "max_width_position_from_nearest_end",
    "base_tip_width_ratio",
]

# CIELAB colour of the seed's pixels: L* 0-100, a* green(-)/red(+), b* blue(-)/yellow(+).
COLOUR_COLUMNS = [
    "colour_L_mean", "colour_L_sd", "colour_a_mean", "colour_a_sd", "colour_b_mean", "colour_b_sd",
]

# Columns produced by measurement, always in pixels at this stage.
MEASURED_COLUMNS = [
    "file_name", "object_id", "detection_score", "AS_seed_area",
    "L_seed_length", "W_seed_width", "LWR_length_to_width_ratio",
    "eccentricity", "solidity", "PL_perimeter_length", "CS_seed_circularity",
    *POSITION_COLUMNS, *SHAPE_PROFILE_COLUMNS, *COLOUR_COLUMNS,
]

# Per-seed metrics in output order: size, shape ratios, profile/asymmetry, colour.
METRIC_ORDER = (
    ["AS_seed_area", "L_seed_length", "W_seed_width", "PL_perimeter_length"]
    + RATIO_COLUMNS + SHAPE_PROFILE_COLUMNS + COLOUR_COLUMNS
)

# Metrics summarised per sample/replicate: per-seed colour SDs are left out.
SUMMARY_METRIC_ORDER = [column for column in METRIC_ORDER if not column.endswith("_sd")]
SUMMARY_STATISTICS = ["mean", "sd", "median", "min", "max"]

# Final per-seed column order (pixel-named; renamed by convert_to_millimetres if --dpi is set).
SEED_COLUMNS = (
    ["file_name", "object_id", "model_index", "detection_score"]
    + METRIC_ORDER + ["is_outlier", "outlier_reason"]
)
REPLICATE_SEED_COLUMNS = (
    ["file_name", "sample_id", "replicate_id", "object_id", "model_index", "detection_score"]
    + METRIC_ORDER + POSITION_COLUMNS + ["is_outlier", "outlier_reason"]
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PREDICTED_MASKS_DIRNAME = "predicted_masks"
SEED_PARAMETERS_FILENAME = "seed_parameters.tsv"
REPLICATES_SUMMARY_FILENAME = "replicates_summary.tsv"
SAMPLES_SUMMARY_FILENAME = "samples_summary.tsv"
EFD_FILENAME = "efd_coefficients.tsv"
RUN_LOG_FILENAME = "run_log.txt"
CONFIG_USED_FILENAME = "config_used.yaml"
