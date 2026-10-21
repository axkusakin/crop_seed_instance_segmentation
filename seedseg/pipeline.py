"""Main processing loop: images in, tables and overlays out."""

import os
import time

import cv2
import numpy as np
import pandas as pd

from .columns import (
    EFD_FILENAME, IMAGE_EXTENSIONS, PREDICTED_MASKS_DIRNAME, REPLICATE_SEED_COLUMNS,
    REPLICATES_SUMMARY_FILENAME, SAMPLES_SUMMARY_FILENAME, SEED_COLUMNS,
    SEED_PARAMETERS_FILENAME,
)
from .measure import measure_instances
from .numbering import renumber_seeds
from .outliers import DEFAULT_K, flag_outliers, mark_no_outliers
from .preprocess import geometric_preprocess, photometric_preprocess
from .replicates import assign_replicates, parse_sample_and_replicates
from .runlog import format_duration
from .summary import summarise
from .units import convert_to_millimetres, summary_metric_columns_for
from .visualize import draw_predictions


def label_replicates(metrics, filename, log):
    """Add sample_id/replicate_id columns to metrics in place.

    Returns (status text, parsed_ok). On a parsing or
    clustering failure, both columns are left blank and a warning is logged.
    """
    try:
        sample_id, replicate_ids = parse_sample_and_replicates(filename)
        metrics["sample_id"] = sample_id
        metrics["replicate_id"] = assign_replicates(metrics, replicate_ids)
    except (ValueError, RuntimeError) as error:
        log.emit(f"Warning: {filename}: {error}. Leaving sample_id/replicate_id blank.")
        metrics["sample_id"] = pd.NA
        metrics["replicate_id"] = pd.NA
        return "replicate parsing FAILED", False

    return f"sample {sample_id!r}, {len(replicate_ids)} replicate(s)", True


def efd_table(metrics, efd, replicates):
    """Prefix one image's EFD coefficients with the seed's identifiers and outlier flag."""
    id_columns = ["file_name"] + (["sample_id", "replicate_id"] if replicates else [])
    keys = metrics[id_columns + ["object_id", "is_outlier"]]
    return keys.merge(efd, on="object_id", how="inner")


def write_summaries(final_df, args, log):
    """Write the summary table(s); returns True if they were created."""
    metric_columns = summary_metric_columns_for(args.dpi)

    if not args.replicates:
        samples_summary = summarise(final_df, ["file_name"], metric_columns)
        samples_path = os.path.join(args.output_dir, SAMPLES_SUMMARY_FILENAME)
        samples_summary.to_csv(samples_path, sep="\t", index=False)
        log.emit(f"Sample summary saved to {samples_path}")
        return True

    labelled_df = final_df.dropna(subset=["sample_id", "replicate_id"])
    if labelled_df.empty:
        log.emit("Warning: no rows had a valid sample_id/replicate_id; summary tables were not created.")
        return False

    replicates_summary = summarise(labelled_df, ["sample_id", "replicate_id"], metric_columns)
    replicates_path = os.path.join(args.output_dir, REPLICATES_SUMMARY_FILENAME)
    replicates_summary.to_csv(replicates_path, sep="\t", index=False)
    log.emit(f"Replicate summary saved to {replicates_path}")

    samples_summary = summarise(
        labelled_df, ["sample_id"], metric_columns, include_replicate_count=True
    )
    samples_path = os.path.join(args.output_dir, SAMPLES_SUMMARY_FILENAME)
    samples_summary.to_csv(samples_path, sep="\t", index=False)
    log.emit(f"Sample summary saved to {samples_path}")
    return True


def process_images(args, model, log):
    predicted_dir = os.path.join(args.output_dir, PREDICTED_MASKS_DIRNAME)
    if args.save_images:
        os.makedirs(predicted_dir, exist_ok=True)

    image_files = [
        filename for filename in sorted(os.listdir(args.input))
        if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        log.fail(f"Error: no supported images found in {args.input}")

    total_images = len(image_files)
    log.emit(f"Found {total_images} image(s) to process in {args.input}")
    log.emit("-" * 78)

    outlier_k = args.outlier_k if args.outlier_k is not None else DEFAULT_K[args.outlier_method]
    tables = []
    efd_tables = []
    skipped_files = []
    parsing_failures = []
    total_seeds_detected = 0
    total_outliers = 0
    run_start = time.perf_counter()

    for index, filename in enumerate(image_files, start=1):
        image_start = time.perf_counter()
        image_path = os.path.join(args.input, filename)
        original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original_bgr is None:
            log.emit(f"[{index}/{total_images}] {filename}: could not read file; skipping.")
            skipped_files.append(filename)
            continue

        # Overlays are drawn on image_bgr; contrast/blur only change the model input.
        image_bgr = geometric_preprocess(original_bgr, args)
        model_input_bgr = photometric_preprocess(image_bgr, args)
        image_rgb = cv2.cvtColor(model_input_bgr, cv2.COLOR_BGR2RGB)
        result = model.detect([image_rgb], verbose=0)[0]
        metrics, instances, raw_instance_count, efd = measure_instances(
            result, filename, image_bgr, args.min_score, args.min_contour_points,
            args.efd_harmonics,
        )

        # Outliers are flagged first so that replicate clustering can ignore them.
        if args.filter_outliers:
            flag_outliers(metrics, args.outlier_method, outlier_k, args.outlier_tail)
        else:
            mark_no_outliers(metrics)

        replicate_by_object_id = {} if args.replicates else None
        replicate_text = "sample/replicate: n/a"
        if not metrics.empty and args.replicates:
            replicate_text, parsed_ok = label_replicates(metrics, filename, log)
            if not parsed_ok:
                parsing_failures.append(filename)

        # Number seeds in reading order (outliers last); overlay labels use the new IDs.
        id_mapping = renumber_seeds(metrics, instances)
        outlier_ids = set(metrics.loc[metrics["is_outlier"], "object_id"])
        total_outliers += len(outlier_ids)
        if not metrics.empty:
            if args.replicates:
                replicate_by_object_id = dict(zip(metrics["object_id"], metrics["replicate_id"]))
            tables.append(metrics)
            if args.efd_harmonics:
                efd["object_id"] = efd["object_id"].map(id_mapping)
                efd_tables.append(efd_table(metrics, efd, args.replicates))

        if args.save_images:
            visualisation = draw_predictions(
                image_bgr, instances, args.alpha, replicate_by_object_id, outlier_ids,
                mark_base=args.mark_base,
            )
            output_name = f"{os.path.splitext(filename)[0]}_predicted.png"
            output_path = os.path.join(predicted_dir, output_name)
            if not cv2.imwrite(output_path, visualisation):
                log.emit(f"Warning: could not write {output_path}")

        elapsed = time.perf_counter() - image_start
        total_seeds_detected += len(instances)

        if instances:
            scores = [instance["score"] for instance in instances]
            score_text = f"mean score {np.mean(scores):.2f} (min {np.min(scores):.2f})"
        else:
            score_text = "no seeds retained"

        avg_time_so_far = (time.perf_counter() - run_start) / index
        remaining_images = total_images - index
        eta_text = format_duration(avg_time_so_far * remaining_images) if remaining_images else "0.0s"
        percent = 100 * index / total_images
        height, width = image_bgr.shape[:2]

        parts = [
            f"[{index}/{total_images}] ({percent:5.1f}%) {filename}: {width}x{height}px",
            f"raw detections {raw_instance_count}",
            f"retained {len(instances)}",
        ]
        if args.filter_outliers:
            parts.append(f"outliers {len(outlier_ids)}")
        if args.replicates:
            parts.append(replicate_text)
        parts += [score_text, f"{elapsed:.2f}s", f"ETA {eta_text}"]
        log.emit(" | ".join(parts))

    total_elapsed = time.perf_counter() - run_start
    log.emit("-" * 78)

    if not tables:
        log.fail("Error: no objects passed the selected score and contour filters. Nothing to save.")

    os.makedirs(args.output_dir, exist_ok=True)

    final_df = pd.concat(tables, ignore_index=True)
    final_df = final_df.sort_values(by=["file_name", "object_id"], kind="stable").reset_index(drop=True)
    final_df = final_df[REPLICATE_SEED_COLUMNS if args.replicates else SEED_COLUMNS]

    if args.dpi is not None:
        final_df = convert_to_millimetres(final_df, args.dpi)
        log.emit(f"Converted pixel measurements to millimetres using {args.dpi} DPI.")

    seed_path = os.path.join(args.output_dir, SEED_PARAMETERS_FILENAME)
    final_df.to_csv(seed_path, sep="\t", index=False)
    log.emit(f"Per-seed table saved to {seed_path}")

    if efd_tables:
        efd_df = pd.concat(efd_tables, ignore_index=True).sort_values(
            by=["file_name", "object_id"], kind="stable"
        )
        efd_path = os.path.join(args.output_dir, EFD_FILENAME)
        efd_df.to_csv(efd_path, sep="\t", index=False)
        log.emit(f"EFD coefficients ({args.efd_harmonics} harmonics) saved to {efd_path}")

    summaries_created = write_summaries(final_df, args, log)

    if args.save_images:
        log.emit(f"Prediction overlays saved in {predicted_dir}")

    images_processed = total_images - len(skipped_files)
    avg_per_image = total_elapsed / images_processed if images_processed else 0.0
    throughput = (images_processed / total_elapsed * 60) if total_elapsed > 0 else 0.0

    log.emit("")
    log.emit("Run summary")
    log.emit(f"  Images processed:      {images_processed}/{total_images}")
    log.emit(f"  Images skipped (I/O):  {len(skipped_files)}"
             + (f" ({', '.join(skipped_files)})" if skipped_files else ""))
    if args.replicates:
        unique_samples = final_df["sample_id"].nunique(dropna=True)
        unique_replicates = (
            final_df.dropna(subset=["sample_id", "replicate_id"])
            .drop_duplicates(subset=["sample_id", "replicate_id"])
            .shape[0]
        )
        log.emit(f"  Replicate parse fails: {len(parsing_failures)}"
                 + (f" ({', '.join(parsing_failures)})" if parsing_failures else ""))
    log.emit(f"  Total seeds kept:      {total_seeds_detected}")
    if args.filter_outliers:
        log.emit(f"  Outliers flagged:      {total_outliers} (excluded from summaries)")
    if args.replicates:
        log.emit(f"  Distinct samples:      {unique_samples}")
        log.emit(f"  Distinct replicates:   {unique_replicates}")
        log.emit(f"  Summary tables built:  {summaries_created}")
    log.emit(f"  Total run time:        {format_duration(total_elapsed)}")
    log.emit(f"  Avg time/image:        {avg_per_image:.2f}s")
    log.emit(f"  Throughput:            {throughput:.1f} images/min")
