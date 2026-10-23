"""Mask R-CNN model construction and the mask-threshold override.

This is the only module that imports TensorFlow and mrcnn.
"""

import os
import sys

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
# Disable the Grappler cost-based optimizations that trigger the CropAndResize warning.
tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

try:
    import mrcnn.model as modellib
    from mrcnn import utils as mrcnn_utils
    from mrcnn.config import Config
except ImportError:
    sys.exit("Error: Mask R-CNN library not found. Run this script from the repository root.")


class InferenceConfig(Config):
    NAME = "seed"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # background + seed
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 8192
    IMAGE_RESIZE_MODE = "pad64"  # as in the original Mask_RCNN.ipynb
    DETECTION_MIN_CONFIDENCE = 0.0
    RPN_NMS_THRESHOLD = 0.4


def install_mask_threshold(threshold):
    """Make model.detect() binarise soft masks at `threshold` instead of 0.5.

    mrcnn.model.MaskRCNN.unmold_detections() calls utils.unmold_mask(), looking
    the function up on the mrcnn.utils module at call time. Replacing that
    module attribute changes the threshold without editing the vendored
    library. The replacement is identical to the original except for the
    threshold value.
    """
    if not hasattr(mrcnn_utils, "unmold_mask") or not hasattr(mrcnn_utils, "resize"):
        sys.exit("Error: mrcnn.utils has no unmold_mask/resize; cannot set the mask threshold.")

    def unmold_mask(mask, bbox, image_shape):
        y1, x1, y2, x2 = bbox
        mask = mrcnn_utils.resize(mask, (y2 - y1, x2 - x1))
        mask = mask >= threshold
        full_mask = np.zeros(image_shape[:2], dtype=bool)
        full_mask[y1:y2, x1:x2] = mask
        return full_mask

    mrcnn_utils.unmold_mask = unmold_mask


def load_model(weights, max_instances, mask_threshold):
    """Build the inference model with the given detection limit and mask threshold."""
    install_mask_threshold(mask_threshold)
    config = InferenceConfig()
    config.DETECTION_MAX_INSTANCES = max_instances
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
    model.load_weights(weights, by_name=True)
    return model
