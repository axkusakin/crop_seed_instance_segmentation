"""Pixel to millimetre conversion."""

from .columns import AREA_COLUMNS, LENGTH_COLUMNS, POSITION_COLUMNS, SUMMARY_METRIC_ORDER


def convert_to_millimetres(df, dpi):
    """Convert pixel-based columns to millimetres, replacing the pixel columns.

    Length-like and position columns scale by mm_per_pixel; area columns
    scale by mm_per_pixel squared. Ratio columns are left untouched. Only
    columns present in df are converted. Returns a new DataFrame with pixel
    columns replaced by their _mm / _mm2 equivalents.
    """
    mm_per_pixel = 25.4 / dpi
    converted = df.copy()

    rename_map = {}
    for column in LENGTH_COLUMNS + POSITION_COLUMNS:
        if column in converted:
            converted[column] = converted[column] * mm_per_pixel
            rename_map[column] = f"{column}_mm"
    for column in AREA_COLUMNS:
        if column in converted:
            converted[column] = converted[column] * (mm_per_pixel ** 2)
            rename_map[column] = f"{column}_mm2"

    return converted.rename(columns=rename_map)


def unit_column_name(column, dpi):
    """Name a pixel-based column takes after conversion (unchanged without --dpi)."""
    if dpi is None:
        return column
    if column in LENGTH_COLUMNS + POSITION_COLUMNS:
        return f"{column}_mm"
    if column in AREA_COLUMNS:
        return f"{column}_mm2"
    return column


def summary_metric_columns_for(dpi):
    """Return the metric columns to summarise, in output order, given the active unit."""
    return [unit_column_name(column, dpi) for column in SUMMARY_METRIC_ORDER]
