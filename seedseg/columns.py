"""Column names and output file names shared across the pipeline."""

# Columns produced by measurement, always in pixels at this stage.
MEASURED_COLUMNS = [
    "file_name", "object_id", "detection_score", "AS_seed_area",
    "L_seed_length", "W_seed_width", "LWR_length_to_width_ratio",
    "eccentricity", "solidity", "PL_perimeter_length", "CS_seed_circularity",
    "centroid_x", "centroid_y",
]

# Final per-seed column order (pixel-named; renamed by convert_to_millimetres if --dpi is set).
SEED_COLUMNS = [
    "file_name", "object_id", "detection_score", "AS_seed_area",
    "L_seed_length", "W_seed_width", "LWR_length_to_width_ratio",
    "eccentricity", "solidity", "PL_perimeter_length", "CS_seed_circularity",
    "is_outlier", "outlier_reason",
]
REPLICATE_SEED_COLUMNS = [
    "file_name", "sample_id", "object_id", "detection_score", "replicate_id",
    "AS_seed_area", "L_seed_length", "W_seed_width", "LWR_length_to_width_ratio",
    "eccentricity", "solidity", "PL_perimeter_length", "CS_seed_circularity",
    "centroid_x", "centroid_y", "is_outlier", "outlier_reason",
]

# Pixel-valued columns and how they scale when converted to millimetres:
# length-like and position columns scale linearly (^1); area columns scale by area (^2).
LENGTH_COLUMNS = ["L_seed_length", "W_seed_width", "PL_perimeter_length"]
AREA_COLUMNS = ["AS_seed_area"]
POSITION_COLUMNS = ["centroid_x", "centroid_y"]

# Dimensionless columns: never converted, never renamed.
RATIO_COLUMNS = [
    "LWR_length_to_width_ratio", "eccentricity", "solidity", "CS_seed_circularity",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PREDICTED_MASKS_DIRNAME = "predicted_masks"
SEED_PARAMETERS_FILENAME = "seed_parameters.tsv"
REPLICATES_SUMMARY_FILENAME = "replicates_summary.tsv"
SAMPLES_SUMMARY_FILENAME = "samples_summary.tsv"
RUN_LOG_FILENAME = "run_log.txt"
CONFIG_USED_FILENAME = "config_used.yaml"
