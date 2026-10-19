"""Command-line interface: argument parsing, validation, and the run wrapper."""

import argparse
import os
import sys
from datetime import datetime

import yaml

from .columns import (
    CONFIG_USED_FILENAME, PREDICTED_MASKS_DIRNAME, REPLICATES_SUMMARY_FILENAME,
    RUN_LOG_FILENAME, SAMPLES_SUMMARY_FILENAME, SEED_PARAMETERS_FILENAME,
)
from .config import ConfigError, load_config, save_config
from .preprocess import FLIP_MODES
from .runlog import RunLog


def build_parser(description):
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config",
                        help=(
                            "YAML config file (see config.example.yaml). Flags given on "
                            "the command line override values from the file."
                        ))
    # -i/-o are required, but may come from the config file, so they're checked after parsing.
    parser.add_argument("-i", "--input", help="Directory containing input images (required)")
    parser.add_argument("-o", "--output-dir",
                        help="Root output directory for all generated tables and images (required)")
    parser.add_argument("-w", "--weights",
                        default="data/barley/model_weights/mask_rcnn_barleyseeds_0040.h5",
                        help="Mask R-CNN weights path")
    parser.add_argument("-r", "--replicates", action=argparse.BooleanOptionalAction, default=False,
                        help=(
                            "Split the seeds of each image into plant replicates using the "
                            "file name (<sample_id>_<r1-r2-...>) and the vertical gaps between "
                            "plants; also writes replicates_summary.tsv"
                        ))
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True,
                        help=(
                            f"Save prediction overlays in <output-dir>/{PREDICTED_MASKS_DIRNAME} "
                            "(default: on; --no-save-images to skip)"
                        ))
    parser.add_argument("--flip", choices=FLIP_MODES, default="none",
                        help=(
                            "Mirror the image before anything else: 'horizontal' (left-right), "
                            "'vertical' (top-bottom) or 'none' (default: none)"
                        ))
    parser.add_argument("--contrast", action=argparse.BooleanOptionalAction, default=False,
                        help="Enhance contrast of the model input with CLAHE on the lightness channel (default: off)")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0,
                        help="CLAHE clip limit; higher values give stronger contrast (default: 2.0)")
    parser.add_argument("--clahe-tile-size", type=int, default=8,
                        help="CLAHE grid size: the image is split into N x N tiles (default: 8)")
    parser.add_argument("--blur", action=argparse.BooleanOptionalAction, default=False,
                        help="Smooth the model input with a Gaussian blur (default: off)")
    parser.add_argument("--blur-kernel", type=int, default=3,
                        help="Gaussian kernel size in pixels, odd (default: 3)")
    parser.add_argument("-cl", "--crop-left", type=float, default=0.0,
                        help="Fraction of the image width removed from the left edge (default: 0.0)")
    parser.add_argument("-cr", "--crop-right", type=float, default=0.0,
                        help="Fraction of the image width removed from the right edge (default: 0.0)")
    parser.add_argument("-ct", "--crop-top", type=float, default=0.0,
                        help="Fraction of the image height removed from the top edge (default: 0.0)")
    parser.add_argument("-cb", "--crop-bottom", type=float, default=0.0,
                        help="Fraction of the image height removed from the bottom edge (default: 0.0)")
    parser.add_argument("--min-score", type=float, default=0.95,
                        help="Minimum detection score (default: 0.95)")
    parser.add_argument("--min-contour-points", type=int, default=100,
                        help="Minimum contour point count (default: 100)")
    parser.add_argument("--mask-threshold", type=float, default=0.1,
                        help=(
                            "Probability threshold used to binarise each soft mask. "
                            "Lower values widen the masks; 0.5 is the "
                            "original Mask R-CNN behaviour (default: 0.1)."
                        ))
    parser.add_argument("--alpha", type=float, default=0.35,
                        help="Mask overlay opacity from 0 to 1 (default: 0.35)")
    parser.add_argument("--max-instances", type=int, default=400,
                        help="Maximum final Mask R-CNN detections per image (default: 400)")
    parser.add_argument("--dpi", type=float, default=None,
                        help=(
                            "Scan resolution in dots per inch. When set, converts all "
                            "length/area/position columns to millimetres "
                            "(mm_per_pixel = 25.4 / dpi) and drops the pixel-based columns "
                            "from the output. Default: None (report pixels)."
                        ))
    return parser


def parse_args(description):
    """Parse flags, layering them over the config file (if any) over the defaults."""
    parser = build_parser(description)
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        try:
            parser.set_defaults(**load_config(pre_args.config))
        except (OSError, ConfigError, yaml.YAMLError) as error:
            parser.error(f"could not load config: {error}")
    args = parser.parse_args()
    if args.input is None:
        parser.error("the input directory is required (-i/--input or 'input' in the config)")
    if args.output_dir is None:
        parser.error("the output directory is required (-o/--output-dir or 'output_dir' in the config)")
    return args


def validate_args(args):
    if not os.path.isdir(args.input):
        sys.exit(f"Error: input directory does not exist: {args.input}")
    if not os.path.isfile(args.weights):
        sys.exit(f"Error: weights file does not exist: {args.weights}")
    for flag, value in (("--crop-left", args.crop_left), ("--crop-right", args.crop_right),
                        ("--crop-top", args.crop_top), ("--crop-bottom", args.crop_bottom)):
        if not 0.0 <= value < 1.0:
            sys.exit(f"Error: {flag} must be at least 0 and less than 1.")
    if args.crop_left + args.crop_right >= 1.0:
        sys.exit("Error: --crop-left and --crop-right together must be less than 1.")
    if args.crop_top + args.crop_bottom >= 1.0:
        sys.exit("Error: --crop-top and --crop-bottom together must be less than 1.")
    if args.flip not in FLIP_MODES:
        sys.exit(f"Error: --flip must be one of {', '.join(FLIP_MODES)}.")
    if args.clahe_clip_limit <= 0:
        sys.exit("Error: --clahe-clip-limit must be a positive number.")
    if args.clahe_tile_size < 1:
        sys.exit("Error: --clahe-tile-size must be at least 1.")
    if args.blur_kernel < 3 or args.blur_kernel % 2 == 0:
        sys.exit("Error: --blur-kernel must be an odd number of at least 3.")
    if not 0.0 < args.mask_threshold < 1.0:
        sys.exit("Error: --mask-threshold must be greater than 0 and less than 1.")
    if not 0.0 <= args.alpha <= 1.0:
        sys.exit("Error: --alpha must be between 0 and 1.")
    if args.max_instances < 1:
        sys.exit("Error: --max-instances must be at least 1.")
    if args.dpi is not None and args.dpi <= 0:
        sys.exit("Error: --dpi must be a positive number.")


def confirm_overwrite(args):
    """Ask before overwriting any existing output; exit if the user declines."""
    table_names = [SEED_PARAMETERS_FILENAME, SAMPLES_SUMMARY_FILENAME]
    if args.replicates:
        table_names.insert(1, REPLICATES_SUMMARY_FILENAME)
    existing_outputs = [
        path for path in (os.path.join(args.output_dir, name) for name in table_names)
        if os.path.exists(path)
    ]
    predicted_dir = os.path.join(args.output_dir, PREDICTED_MASKS_DIRNAME)
    if args.save_images and os.path.isdir(predicted_dir) and os.listdir(predicted_dir):
        existing_outputs.append(predicted_dir)
    if existing_outputs:
        response = input(
            f"Warning: {', '.join(existing_outputs)} already exist(s). Overwrite? (y/n): "
        )
        if response.lower() != "y":
            sys.exit("Process aborted by user.")


def log_parameters(args, log):
    log.emit(f"Run started:          {datetime.now().isoformat(timespec='seconds')}")
    log.emit(f"Config file:          {args.config or 'none'}")
    log.emit(f"Input directory:      {args.input}")
    log.emit(f"Output directory:     {args.output_dir}")
    log.emit(f"Weights:              {args.weights}")
    log.emit(f"Replicates:           {args.replicates}")
    log.emit(f"Flip:                 {args.flip}")
    log.emit(f"Crop (L/R/T/B):       {args.crop_left}/{args.crop_right}/{args.crop_top}/{args.crop_bottom}")
    contrast_text = (f"CLAHE clip {args.clahe_clip_limit}, tiles {args.clahe_tile_size}"
                     if args.contrast else "off")
    log.emit(f"Contrast:             {contrast_text}")
    log.emit(f"Blur:                 {f'Gaussian, kernel {args.blur_kernel}' if args.blur else 'off'}")
    log.emit(f"Min score:            {args.min_score}")
    log.emit(f"Min contour points:   {args.min_contour_points}")
    log.emit(f"Mask threshold:       {args.mask_threshold}")
    log.emit(f"Max instances:        {args.max_instances}")
    log.emit(f"DPI (unit conversion):{'pixels (none)' if args.dpi is None else args.dpi}")
    log.emit(f"Save overlays:        {args.save_images}")
    log.emit("=" * 78)


def main(description=None):
    args = parse_args(description)
    validate_args(args)
    confirm_overwrite(args)

    log = RunLog()
    log_parameters(args, log)
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, CONFIG_USED_FILENAME)
    save_config(args, config_path)
    log.emit(f"Resolved parameters saved to {config_path}")

    try:
        # Imported here so that --help and argument errors don't wait for TensorFlow.
        from .model import load_model
        from .pipeline import process_images

        log.emit(f"Loading weights from {args.weights}")
        model = load_model(args.weights, args.max_instances, args.mask_threshold)
        process_images(args, model, log)
    finally:
        # Written even when the run fails, so failed runs leave a record too.
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, RUN_LOG_FILENAME)
        log.write(log_path)
        print(f"Run log saved to {log_path}")
