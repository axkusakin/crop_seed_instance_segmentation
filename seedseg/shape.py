"""Shape-profile, asymmetry, colour and Fourier descriptors of a single seed.

All functions take the seed's outer contour (the same largest connected
component used for the basic metrics) and work on the seed filled into a
local canvas the size of its bounding box.

Orientation. The seed is rotated onto its principal (long) axis u, with the
perpendicular offset v measured from the chord joining the seed's two ends
(not from the centroid, which shifts towards a bulging side). The width profile w(u) is measured in PROFILE_BINS
slices along the length. The blunter end -- the one with the larger mean
width over the outer END_FRACTION of the length -- is taken as the base, and
u runs from base (0) to tip (1). base_tip_width_ratio (>= 1) says how
distinct the two ends are: near 1, the orientation is unreliable.

Metrics (all dimensionless):
  lateral_area_asymmetry       |A(v>0) - A(v<0)| / A, across the long axis; 0 = symmetric
  lateral_mirror_iou           overlap of the seed with its mirror image across the
                               long axis; 1 = symmetric
  longitudinal_area_asymmetry  (A_base half - A_tip half) / A, signed
  width_profile_skewness       skewness of w(u) along the length, base -> tip, signed
  max_width_position           position of the maximum width, 0 = base, 1 = tip
  max_width_position_from_nearest_end
                               min(p, 1 - p): orientation-free version, 0..0.5
  base_tip_width_ratio         mean width near the base / near the tip, >= 1
"""

import cv2
import numpy as np
from pyefd import elliptic_fourier_descriptors

PROFILE_BINS = 50
END_FRACTION = 0.15
# Pixels within this distance (px) of each extreme along the long axis locate its end points.
# Averaging over a wider window biases the chord towards a bulging side.
ENDPOINT_TOLERANCE = 0.5



def fill_contour(contour):
    """Return (filled uint8 mask of the bounding box, (x0, y0) offset of that box)."""
    x0, y0, width, height = cv2.boundingRect(contour)
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(canvas, [contour - np.array([x0, y0])], -1, 1, thickness=cv2.FILLED)
    return canvas, (x0, y0)


def shape_profile(contour):
    """Return (metrics dict, base point (x, y) in image coordinates)."""
    canvas, (x0, y0) = fill_contour(contour)
    ys, xs = np.nonzero(canvas)
    points = np.column_stack([xs + x0, ys + y0]).astype(float)
    centre = points.mean(axis=0)
    centred = points - centre

    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centred, rowvar=False))
    major = eigenvectors[:, np.argmax(eigenvalues)]
    minor = np.array([-major[1], major[0]])
    u = centred @ major
    v = centred @ minor

    u_min, u_max = u.min(), u.max()
    span = max(u_max - u_min, 1e-9)

    # Measure v from the chord joining the two ends, not from the centroid:
    # the centroid shifts towards a bulging side and would hide the asymmetry.
    start_tip = u <= u_min + ENDPOINT_TOLERANCE
    end_tip = u >= u_max - ENDPOINT_TOLERANCE
    v_start, v_end = v[start_tip].mean(), v[end_tip].mean()
    v = v - (v_start + (v_end - v_start) * (u - u_min) / span)
    bins = np.clip(((u - u_min) / span * PROFILE_BINS).astype(int), 0, PROFILE_BINS - 1)
    top = np.full(PROFILE_BINS, np.nan)
    bottom = np.full(PROFILE_BINS, np.nan)
    np.fmax.at(top, bins, v)
    np.fmin.at(bottom, bins, v)
    filled = ~np.isnan(top)
    width = np.where(filled, top - bottom + 1.0, 0.0)

    # Orient base -> tip: the blunter (wider) end is the base.
    end_bins = max(1, int(round(END_FRACTION * PROFILE_BINS)))
    start_width, end_width = width[:end_bins].mean(), width[-end_bins:].mean()
    base_at_end = end_width > start_width
    if base_at_end:
        width = width[::-1]
        top, bottom, filled = top[::-1], bottom[::-1], filled[::-1]
        u_along = u_max - u
    else:
        u_along = u - u_min
    base_u = u_max if base_at_end else u_min
    base_point = centre + major * base_u

    blunt, sharp = max(start_width, end_width), min(start_width, end_width)
    base_tip_width_ratio = blunt / sharp if sharp > 0 else np.nan

    area = float(len(u))
    lateral_area_asymmetry = abs(np.count_nonzero(v > 0) - np.count_nonzero(v < 0)) / area

    # Mirror IoU per slice: interval [bottom, top] vs its mirror [-top, -bottom].
    t, b = top[filled], bottom[filled]
    overlap = np.clip(np.minimum(t, -b) - np.maximum(b, -t), 0, None)
    union = 2 * (t - b) - overlap
    lateral_mirror_iou = overlap.sum() / union.sum() if union.sum() > 0 else np.nan

    base_half = np.count_nonzero(u_along < span / 2)
    longitudinal_area_asymmetry = (base_half - (area - base_half)) / area

    positions = (np.arange(PROFILE_BINS) + 0.5) / PROFILE_BINS
    weights = width / width.sum()
    mean_position = (weights * positions).sum()
    sd_position = np.sqrt((weights * (positions - mean_position) ** 2).sum())
    width_profile_skewness = (
        (weights * (positions - mean_position) ** 3).sum() / sd_position ** 3
        if sd_position > 0 else np.nan
    )

    smoothed = np.convolve(width, np.ones(3) / 3, mode="same")
    max_width_position = float(positions[np.argmax(smoothed)])

    metrics = {
        "lateral_area_asymmetry": lateral_area_asymmetry,
        "lateral_mirror_iou": lateral_mirror_iou,
        "longitudinal_area_asymmetry": longitudinal_area_asymmetry,
        "width_profile_skewness": width_profile_skewness,
        "max_width_position": max_width_position,
        "max_width_position_from_nearest_end": min(max_width_position, 1 - max_width_position),
        "base_tip_width_ratio": base_tip_width_ratio,
    }
    return metrics, (float(base_point[0]), float(base_point[1]))


def colour_lab(image_bgr, contour):
    """Mean and SD of CIELAB L* (0-100), a*, b* over the seed's pixels."""
    canvas, (x0, y0) = fill_contour(contour)
    height, width = canvas.shape
    crop = image_bgr[y0:y0 + height, x0:x0 + width]
    lab = cv2.cvtColor(crop.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    pixels = lab[canvas.astype(bool)]
    means, sds = pixels.mean(axis=0), pixels.std(axis=0)
    return {
        "colour_L_mean": float(means[0]), "colour_L_sd": float(sds[0]),
        "colour_a_mean": float(means[1]), "colour_a_sd": float(sds[1]),
        "colour_b_mean": float(means[2]), "colour_b_sd": float(sds[2]),
    }


def efd_column_names(order):
    """Names of the normalised EFD coefficients, without the 3 constant ones."""
    names = [f"efd_{harmonic}_{letter}" for harmonic in range(1, order + 1) for letter in "abcd"]
    return names[3:]


def efd_coefficients(contour, order):
    """Normalised elliptic Fourier coefficients (size, rotation and start-point invariant).

    After normalisation the first harmonic is always (1, 0, 0, d), so its
    first three coefficients are dropped.
    """
    coefficients = elliptic_fourier_descriptors(contour.reshape(-1, 2), order=order, normalize=True)
    return coefficients.flatten()[3:]
