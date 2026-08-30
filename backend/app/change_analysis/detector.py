"""
Change detector for bi-temporal satellite imagery.

Uses perceptual colour-space differencing (CIE LAB) with Gaussian noise
suppression and Otsu thresholding.  The architecture is designed so a
dedicated deep-learning change-detection model (e.g. BIT, ChangeFormer)
can be swapped in later by subclassing or replacing ``detect()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger("satquery.change.detector")


@dataclass
class DetectionResult:
    """Raw output of the change detector before region extraction."""
    change_mask: np.ndarray       # (H, W) bool — True where change detected
    magnitude_map: np.ndarray     # (H, W) float32 in [0, 1]
    changed_area_fraction: float  # 0..1


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an (H, W, 3) float32 RGB [0,1] image to CIE LAB.

    Uses a simplified sRGB→XYZ→LAB pipeline.  Perceptual uniformity of LAB
    makes the Euclidean distance a much better proxy for *visual* change
    than raw RGB differences.
    """

    # --- sRGB linearisation ---
    linear = np.where(rgb <= 0.04045, rgb / 12.92,
                      ((rgb + 0.055) / 1.055) ** 2.4)

    # --- Linear RGB → XYZ (D65 illuminant) ---
    # Using the standard sRGB to XYZ matrix
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # --- Normalise by D65 white point ---
    x /= 0.95047
    # y already normalised (Yn = 1.0)
    z /= 1.08883

    # --- XYZ → LAB ---
    epsilon = 0.008856
    kappa = 903.3

    def _f(t: np.ndarray) -> np.ndarray:
        return np.where(t > epsilon, np.cbrt(t), (kappa * t + 16) / 116)

    fx, fy, fz = _f(x), _f(y), _f(z)

    L = 116 * fy - 16       # 0..100
    a = 500 * (fx - fy)     # roughly -128..128
    b_ch = 200 * (fy - fz)  # roughly -128..128

    return np.stack([L, a, b_ch], axis=-1)


def _gaussian_blur(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian blur.  Uses scipy if available, otherwise falls back
    to a simple box-filter approximation (still good enough for noise
    suppression).
    """
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=sigma)
    except ImportError:
        # Box-filter approximation (3-pass is a reasonable Gaussian approx)
        kernel_size = max(3, int(sigma * 3) | 1)
        from numpy.lib.stride_tricks import sliding_window_view
        pad = kernel_size // 2
        padded = np.pad(image, pad, mode="reflect")
        # Simple uniform 1-D convolutions along each axis
        for _ in range(3):
            cumsum = np.cumsum(padded, axis=0)
            padded = (cumsum[kernel_size:] - cumsum[:-kernel_size]) / kernel_size
            cumsum = np.cumsum(padded, axis=1)
            padded = (cumsum[:, kernel_size:] - cumsum[:, :-kernel_size]) / kernel_size
        return padded[:image.shape[0], :image.shape[1]]


def _otsu_threshold(data: np.ndarray) -> float:
    """Compute the Otsu threshold for a 1-D array of float values in [0,1]."""
    hist, bin_edges = np.histogram(data.ravel(), bins=256, range=(0, 1))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return 0.5

    sum_total = (hist * bin_centres).sum()
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    threshold = 0.5

    for i in range(256):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += hist[i] * bin_centres[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = bin_centres[i]

    return float(threshold)


class ChangeDetector:
    """
    Pixel-level change detector using perceptual colour-space differencing.

    This is a practical *baseline* detector — not a trained deep model.
    The class is designed so that a dedicated CD model can replace the
    ``detect()`` method later without changing the rest of the pipeline.
    """

    def __init__(
        self,
        blur_sigma: float = 2.0,
        min_threshold: float = 0.05,
    ):
        self.blur_sigma = blur_sigma
        self.min_threshold = min_threshold

    def detect(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> DetectionResult:
        """
        Detect changes between two co-registered images.

        Parameters
        ----------
        before, after : (H, W, 3) float32 arrays in [0, 1]

        Returns
        -------
        DetectionResult
        """
        assert before.shape == after.shape, (
            f"Shape mismatch: {before.shape} vs {after.shape}"
        )

        # 1. Convert to LAB for perceptually meaningful differences
        lab_before = _rgb_to_lab(before)
        lab_after = _rgb_to_lab(after)

        # 2. Euclidean distance in LAB space
        diff = np.sqrt(np.sum((lab_before - lab_after) ** 2, axis=-1))

        # 3. Normalise to [0, 1]
        max_diff = diff.max()
        if max_diff > 0:
            diff = diff / max_diff
        else:
            # Identical images
            return DetectionResult(
                change_mask=np.zeros(before.shape[:2], dtype=bool),
                magnitude_map=diff,
                changed_area_fraction=0.0,
            )

        # 4. Gaussian blur to suppress noise
        diff_smooth = _gaussian_blur(diff, sigma=self.blur_sigma)
        diff_smooth = np.clip(diff_smooth, 0, 1)

        # 5. Otsu thresholding
        threshold = max(self.min_threshold, _otsu_threshold(diff_smooth))
        change_mask = diff_smooth > threshold

        # 6. Changed area fraction
        changed_area_fraction = float(change_mask.sum()) / change_mask.size

        logger.info(
            "Detection complete: threshold=%.3f, changed=%.2f%%",
            threshold,
            changed_area_fraction * 100,
        )

        return DetectionResult(
            change_mask=change_mask,
            magnitude_map=diff_smooth,
            changed_area_fraction=changed_area_fraction,
        )
