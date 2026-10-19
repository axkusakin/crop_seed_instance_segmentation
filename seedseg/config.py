"""YAML run configuration.

A config file sets the same parameters as the command-line flags, grouped
into sections. Precedence: built-in defaults < config file < explicit flags.
The resolved parameters of every run are saved as config_used.yaml.
"""

import yaml

# YAML key path -> argparse dest. This is the full schema of the config file.
CONFIG_KEYS = {
    ("input",): "input",
    ("output_dir",): "output_dir",
    ("replicates",): "replicates",
    ("preprocessing", "crop", "left"): "crop_left",
    ("preprocessing", "crop", "right"): "crop_right",
    ("preprocessing", "crop", "top"): "crop_top",
    ("preprocessing", "crop", "bottom"): "crop_bottom",
    ("model", "weights"): "weights",
    ("model", "min_score"): "min_score",
    ("model", "min_contour_points"): "min_contour_points",
    ("model", "mask_threshold"): "mask_threshold",
    ("model", "max_instances"): "max_instances",
    ("output", "dpi"): "dpi",
    ("output", "save_images"): "save_images",
    ("output", "alpha"): "alpha",
}

# Expected Python types per dest; ints are accepted where floats are expected.
DEST_TYPES = {
    "input": str, "output_dir": str, "weights": str,
    "replicates": bool, "save_images": bool,
    "crop_left": float, "crop_right": float, "crop_top": float, "crop_bottom": float,
    "min_score": float, "mask_threshold": float, "alpha": float, "dpi": float,
    "min_contour_points": int, "max_instances": int,
}

NULLABLE_DESTS = {"dpi"}


class ConfigError(ValueError):
    pass


def _flatten(node, prefix=()):
    """Yield (key path, value) for every leaf of a nested mapping."""
    for key, value in node.items():
        path = prefix + (str(key),)
        if isinstance(value, dict):
            yield from _flatten(value, path)
        else:
            yield path, value


def _check_type(dest, value, path_text):
    expected = DEST_TYPES[dest]
    if value is None:
        if dest in NULLABLE_DESTS:
            return None
        raise ConfigError(f"{path_text}: value must not be empty")
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if expected is int and isinstance(value, bool):
        raise ConfigError(f"{path_text}: expected an integer, got {value!r}")
    if not isinstance(value, expected):
        raise ConfigError(f"{path_text}: expected {expected.__name__}, got {value!r}")
    return value


def load_config(path):
    """Read a YAML config and return {argparse dest: value}."""
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping of sections")

    values = {}
    for key_path, value in _flatten(data):
        path_text = ".".join(key_path)
        dest = CONFIG_KEYS.get(key_path)
        if dest is None:
            raise ConfigError(f"{path}: unknown config key '{path_text}'")
        values[dest] = _check_type(dest, value, path_text)
    return values


def to_config_dict(args):
    """Rebuild the nested config structure from resolved arguments."""
    config = {}
    for key_path, dest in CONFIG_KEYS.items():
        node = config
        for key in key_path[:-1]:
            node = node.setdefault(key, {})
        node[key_path[-1]] = getattr(args, dest)
    return config


def save_config(args, path):
    with open(path, "w") as handle:
        yaml.safe_dump(to_config_dict(args), handle, sort_keys=False, allow_unicode=True)
