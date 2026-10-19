"""Image manipulations applied before inference."""


def crop_edges(image_bgr, left, right, top, bottom):
    """Remove a fraction of the original width (left/right) or height (top/bottom)."""
    height, width = image_bgr.shape[:2]
    left_px = int(round(width * left))
    right_px = int(round(width * right))
    top_px = int(round(height * top))
    bottom_px = int(round(height * bottom))
    return image_bgr[top_px:height - bottom_px, left_px:width - right_px]
