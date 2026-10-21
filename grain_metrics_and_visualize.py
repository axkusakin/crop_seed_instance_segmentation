#!/usr/bin/env python3
"""Detect barley grains, save per-seed metrics, summary statistics, and
mask overlays -- all under one output folder.

One Mask R-CNN inference pass per image. Every predicted object that passes
--min-score and --min-contour-points is written to the per-seed table and
drawn in the overlay (unless --no-save-images). The overlay label is the
exact object_id in the table.

Seeds are numbered from 1 in reading order -- in rows from top to bottom,
left to right within a row -- with outliers numbered last. The numbering is
continuous across all replicates of an image; the original Mask R-CNN index
is kept in model_index.

Outliers
--------
A seed whose area or length is an outlier among the seeds of its image (IQR
or MAD rule, --outlier-method) is flagged with is_outlier/outlier_reason. It
stays in seed_parameters.tsv and is drawn in red, but is excluded from the
summary tables and from replicate clustering. --no-filter-outliers disables it.

Replicates
----------
By default every seed in an image belongs to one accession (file_name). With
--replicates, the seeds are split into plant replicates using the file name:

    SV1_18-19_015_1-2-3.jpg  -> sample_id "SV1_18-19_015", replicate IDs [1, 2, 3]
    SV_022_5-4-8.jpg         -> sample_id "SV_022",        replicate IDs [5, 4, 8]
    SV_007_4.jpg             -> sample_id "SV_007",        replicate ID  [4] (single plant)

Plants must be laid out vertically (one above another) with an empty gap
between them. Seed centroids are connected into a minimum spanning tree
(MST); removing the (k - 1) longest edges always yields exactly k groups.
Groups are ordered top to bottom by mean centroid y and matched, in that
order, to the replicate IDs in the file name.

Units
-----
By default all length/area/position measurements are in pixels. Pass --dpi
to convert them to millimetres (mm_per_pixel = 25.4 / dpi); the tables then
report ONLY the _mm / _mm2 columns plus the dimensionless ratios.

Mask threshold
--------------
Mask R-CNN predicts a soft (0-1) mask for every detected seed, which
mrcnn.utils.unmold_mask binarises at a fixed 0.5. --mask-threshold overrides
that value: lower values move the mask border outwards, higher values move it
inwards. The default is 0.1; pass 0.5 to reproduce the original Mask R-CNN
behaviour. The mask can never extend beyond the detection box.

Output structure
----------------
    <output-dir>/
        seed_parameters.tsv      one row per detected seed
        samples_summary.tsv      summary stats per file_name (or sample_id with --replicates)
        replicates_summary.tsv   summary stats per (sample_id, replicate_id), --replicates only
        run_log.txt              run parameters and a per-run summary
        config_used.yaml         resolved parameters (reusable with --config)
        predicted_masks/         overlays, skipped with --no-save-images

Config file
-----------
All parameters can be kept in a YAML file passed with --config (see
config.example.yaml). Flags given on the command line override the file.

Examples
--------
    # Tables only, pixels
    python grain_metrics_and_visualize.py -i images -o output_dir

    # All parameters from a config file, overriding one of them
    python grain_metrics_and_visualize.py --config my_config.yaml --mask-threshold 0.3

    # Per-plant replicates, overlays, millimetres (500 DPI scan)
    python grain_metrics_and_visualize.py -i images -o output_dir --replicates \\
        --dpi 500

    # Trim 8% from both lateral edges, mask threshold 0.35
    python grain_metrics_and_visualize.py -i images -o output_dir \\
        -cl 0.08 -cr 0.08 --mask-threshold 0.35
"""

from seedseg.cli import main

if __name__ == "__main__":
    main(description=__doc__)
