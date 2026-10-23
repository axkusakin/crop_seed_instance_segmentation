"""Image manipulations applied before inference.

Geometric steps (flip, crop) change both the model input and the image the
overlays are drawn on, so object coordinates always refer to the corrected
image. Photometric steps (contrast, blur) are applied to the model input
only; overlays show the unmodified colours.

Order: flip -> crop -> contrast -> blur. Because the flip comes first, the
crop sides refer to the flipped (corrected) image.
"""

import cv2

FLIP_MODES = ("none", "horizontal", "vertical")
_FLIP_CODES = {"horizontal": 1, "vertical": 0}


def flip_image(image_bgr, mode):
    """Mirror the image left-right ("horizontal") or top-bottom ("vertical")."""
    if mode == "none":
        return image_bgr
    return cv2.flip(image_bgr, _FLIP_CODES[mode])


def crop_edges(image_bgr, left, right, top, bottom):
    """Remove a fraction of the original width (left/right) or height (top/bottom)."""
    height, width = image_bgr.shape[:2]
    left_px = int(round(width * left))
    right_px = int(round(width * right))
    top_px = int(round(height * top))
    bottom_px = int(round(height * bottom))
    return image_bgr[top_px:height - bottom_px, left_px:width - right_px]


def enhance_contrast(image_bgr, clip_limit, tile_size):
    """Apply CLAHE to the lightness channel (LAB), leaving colours unchanged."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab = cv2.merge((clahe.apply(lightness), a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gaussian_blur(image_bgr, kernel_size):
    """Smooth the image with a square Gaussian kernel (odd size, sigma from size)."""
    return cv2.GaussianBlur(image_bgr, (kernel_size, kernel_size), 0)


def geometric_preprocess(image_bgr, args):
    """Flip and crop: the result is both the overlay canvas and the model input base."""
    image_bgr = flip_image(image_bgr, args.flip)
    return crop_edges(image_bgr, args.crop_left, args.crop_right, args.crop_top, args.crop_bottom)


def photometric_preprocess(image_bgr, args):
    """Contrast and blur: applied to the model input only."""
    if args.contrast:
        image_bgr = enhance_contrast(image_bgr, args.clahe_clip_limit, args.clahe_tile_size)
    if args.blur:
        image_bgr = gaussian_blur(image_bgr, args.blur_kernel)
    return image_bgr
