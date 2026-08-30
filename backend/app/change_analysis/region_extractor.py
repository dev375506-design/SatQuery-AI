"""
Extract and characterise significant change regions from the binary
change mask produced by the detector.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from app.change_analysis.schemas import BoundingBox, ChangeRegion

logger = logging.getLogger("satquery.change.region_extractor")


def _location_description(cx: float, cy: float) -> str:
    """
    Describe the approximate location of a region within the image
    using quadrant names (e.g. "upper-left", "center").

    cx, cy are normalised coordinates in [0, 1].
    """
    v = "upper" if cy < 0.33 else ("lower" if cy > 0.66 else "central")
    h = "left" if cx < 0.33 else ("right" if cx > 0.66 else "central")
    if v == "central" and h == "central":
        return "center of the scene"
    if v == "central":
        return f"{h} portion of the scene"
    if h == "central":
        return f"{v}-central portion of the scene"
    return f"{v}-{h} portion of the scene"


def _connected_components(mask: np.ndarray) -> np.ndarray:
    """
    Label connected components in a 2-D boolean mask.

    Uses scipy.ndimage if available, otherwise falls back to a simple
    flood-fill implementation.
    """
    try:
        from scipy.ndimage import label as scipy_label
        labels, _ = scipy_label(mask)
        return labels
    except ImportError:
        pass

    # --- Fallback: simple BFS flood fill ---
    labels = np.zeros_like(mask, dtype=np.int32)
    current_label = 0
    h, w = mask.shape

    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == 0:
                current_label += 1
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if (
                        cy < 0 or cy >= h or cx < 0 or cx >= w
                        or not mask[cy, cx]
                        or labels[cy, cx] != 0
                    ):
                        continue
                    labels[cy, cx] = current_label
                    stack.extend([
                        (cy - 1, cx), (cy + 1, cx),
                        (cy, cx - 1), (cy, cx + 1),
                    ])

    return labels


def extract_regions(
    change_mask: np.ndarray,
    magnitude_map: np.ndarray,
    min_area_fraction: float = 0.001,
    max_regions: int = 20,
) -> List[ChangeRegion]:
    """
    Extract change regions from the binary mask.

    Parameters
    ----------
    change_mask : (H, W) bool
    magnitude_map : (H, W) float32 [0, 1]
    min_area_fraction : minimum fraction of total image area for a region
                        to be kept (filters noise)
    max_regions : cap on the number of regions returned

    Returns
    -------
    List of ChangeRegion, sorted by area (largest first).
    """
    h, w = change_mask.shape
    total_pixels = h * w
    min_area = int(total_pixels * min_area_fraction)

    labels = _connected_components(change_mask)
    region_ids = np.unique(labels)
    region_ids = region_ids[region_ids > 0]  # skip background (0)

    regions: List[ChangeRegion] = []

    for rid in region_ids:
        region_mask = labels == rid
        area = int(region_mask.sum())
        if area < min_area:
            continue

        ys, xs = np.where(region_mask)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        # Centre of mass (normalised)
        cx = float(xs.mean()) / w
        cy = float(ys.mean()) / h

        mean_mag = float(magnitude_map[region_mask].mean())

        # Confidence heuristic: combination of magnitude and area
        conf = min(1.0, 0.5 + mean_mag * 0.3 + (area / total_pixels) * 2.0)

        regions.append(ChangeRegion(
            region_id=int(rid),
            bbox=BoundingBox(
                x=x_min,
                y=y_min,
                width=x_max - x_min + 1,
                height=y_max - y_min + 1,
            ),
            area_pixels=area,
            area_fraction=area / total_pixels,
            location_description=_location_description(cx, cy),
            mean_magnitude=mean_mag,
            confidence=round(conf, 3),
        ))

    # Sort by area descending
    regions.sort(key=lambda r: r.area_pixels, reverse=True)

    if len(regions) > max_regions:
        regions = regions[:max_regions]

    logger.info(
        "Extracted %d significant change regions (filtered from %d total)",
        len(regions),
        len(region_ids),
    )

    return regions
