"""Pixel to millimetre conversion."""

from .columns import AREA_COLUMNS, LENGTH_COLUMNS, POSITION_COLUMNS, RATIO_COLUMNS


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


def shape_metric_columns_for(dpi):
    """Return the shape-metric column names to summarise, given the active unit."""
    if dpi is None:
        return LENGTH_COLUMNS + AREA_COLUMNS + RATIO_COLUMNS
    length_mm = [f"{column}_mm" for column in LENGTH_COLUMNS]
    area_mm2 = [f"{column}_mm2" for column in AREA_COLUMNS]
    return length_mm + area_mm2 + RATIO_COLUMNS
