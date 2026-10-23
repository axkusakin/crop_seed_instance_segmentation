"""Sample sheet: grain type (hulled / naked) per sample, used to orient each seed.

The sheet is a delimited text file (tab, comma or semicolon) with one row per
sample: an ID column (default 'sample_id') and a 'grain_hull' column, 1 for
hulled and 0 for naked barley.

Which end of a seed is its base depends on the grain type. In hulled barley
the base is the narrow end -- the attachment to the rachis pinches the
outline -- and the tip is widened. In naked barley the base is the blunter end.
Samples without a known type are oriented like naked barley (blunt end).
"""

import os

import pandas as pd

from .replicates import parse_sample_and_replicates

GRAIN_HULL_COLUMN = "grain_hull"
GRAIN_TYPES = {1: "hulled", 0: "naked"}
BASE_END_BY_GRAIN_TYPE = {"hulled": "narrow", "naked": "blunt"}
DEFAULT_BASE_END = "blunt"


class SampleSheetError(ValueError):
    pass


def load_sample_sheet(path, id_column):
    """Read the sheet and return {sample_id: 'hulled' | 'naked'}."""
    try:
        sheet = pd.read_csv(path, sep=None, engine="python", dtype=str, skipinitialspace=True)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise SampleSheetError(f"{path}: could not read the sample sheet: {error}") from error
    sheet.columns = sheet.columns.str.strip()

    missing = [column for column in (id_column, GRAIN_HULL_COLUMN) if column not in sheet]
    if missing:
        raise SampleSheetError(
            f"{path}: missing column(s) {', '.join(missing)} "
            f"(found: {', '.join(sheet.columns)})"
        )

    sheet = sheet[[id_column, GRAIN_HULL_COLUMN]].apply(lambda column: column.str.strip())
    sheet = sheet.dropna(subset=[id_column])
    grain_types = {}
    for row_number, (sample_id, hull) in enumerate(sheet.itertuples(index=False), start=2):
        if hull not in ("0", "1"):
            raise SampleSheetError(
                f"{path}, row {row_number}: {GRAIN_HULL_COLUMN} must be 1 (hulled) or 0 (naked), "
                f"got {hull!r} for sample {sample_id!r}"
            )
        grain_type = GRAIN_TYPES[int(hull)]
        if grain_types.get(sample_id, grain_type) != grain_type:
            raise SampleSheetError(f"{path}: sample {sample_id!r} is listed as both hulled and naked")
        grain_types[sample_id] = grain_type
    return grain_types


def grain_type_for(file_name, grain_types):
    """Look up an image's grain type in the sheet.

    Tried in turn: the parsed sample_id (SV3_19-20_SV002 for
    SV3_19-20_SV002_1-5-6.tif), its last underscore-separated part -- the
    sample number (SV002) -- and the bare file name. Returns None when the
    sample is not in the sheet.
    """
    candidates = []
    try:
        sample_id = parse_sample_and_replicates(file_name)[0]
        candidates += [sample_id, sample_id.rpartition("_")[2]]
    except ValueError:
        pass
    candidates.append(os.path.splitext(file_name)[0])
    for candidate in candidates:
        if candidate in grain_types:
            return grain_types[candidate]
    return None


def base_end_for(grain_type):
    """'narrow' or 'blunt': which end of the seed is its base."""
    return BASE_END_BY_GRAIN_TYPE.get(grain_type, DEFAULT_BASE_END)
