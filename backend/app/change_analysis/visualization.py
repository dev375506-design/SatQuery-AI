"""
Visualisation utilities for bi-temporal change analysis.

Generates:
    1. Change map — heatmap of the difference magnitude
    2. Change overlay — after image with change regions highlighted
    3. Region boxes — optional bounding-box visualisation
"""

from __future__ import annotations

import io
import base64
import logging
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.change_analysis.schemas import ChangeRegion

logger = logging.getLogger("satquery.change.visualization")


def _magnitude_to_heatmap(magnitude: np.ndarray) -> np.ndarray:
    """
    Convert a [0,1] magnitude map to an RGB heatmap.

    Colour ramp:  black → dark red → red → orange → yellow → white
    """
    h, w = magnitude.shape
    out = np.zeros((h, w, 3), dtype=np.float32)

    # Apply a non-linear curve to enhance visibility of moderate changes
    mag = np.clip(magnitude, 0, 1) ** 0.7

    # R channel: rises early
    out[..., 0] = np.clip(mag * 3.0, 0, 1)
    # G channel: rises later
    out[..., 1] = np.clip(mag * 3.0 - 1.0, 0, 1)
    # B channel: only at very high magnitudes (white hot)
    out[..., 2] = np.clip(mag * 3.0 - 2.0, 0, 1)

    return out


def generate_change_map(magnitude_map: np.ndarray) -> str:
    """
    Generate a heatmap visualisation of change magnitude.

    Returns a base64-encoded PNG string.
    """
    heatmap = _magnitude_to_heatmap(magnitude_map)
    img = Image.fromarray((heatmap * 255).astype(np.uint8))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_change_overlay(
    after_image: np.ndarray,
    change_mask: np.ndarray,
    magnitude_map: np.ndarray,
    regions: List[ChangeRegion],
    overlay_alpha: float = 0.45,
) -> str:
    """
    Generate the after image with change regions highlighted in
    semi-transparent magenta / red, plus labelled bounding boxes.

    Returns a base64-encoded PNG string.
    """
    h, w = after_image.shape[:2]

    # Start from the after image
    overlay = after_image.copy()

    # Create the highlight colour layer (magenta where changed)
    highlight = np.zeros_like(overlay)
    highlight[change_mask, 0] = 1.0    # R
    highlight[change_mask, 1] = 0.15   # G
    highlight[change_mask, 2] = 0.55   # B

    # Blend: stronger highlighting where magnitude is higher
    alpha = np.zeros((h, w, 1), dtype=np.float32)
    alpha[change_mask, 0] = (
        overlay_alpha * np.clip(magnitude_map[change_mask], 0.3, 1.0)
    )

    overlay = overlay * (1 - alpha) + highlight * alpha
    overlay = np.clip(overlay, 0, 1)

    # Convert to PIL for drawing bounding boxes and labels
    img = Image.fromarray((overlay * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    for region in regions:
        bb = region.bbox
        x1, y1 = bb.x, bb.y
        x2, y2 = bb.x + bb.width, bb.y + bb.height

        # Draw bounding box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 80, 120),
            width=2,
        )

        # Label
        label = f"R{region.region_id} ({region.confidence*100:.0f}%)"
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (IOError, OSError):
            font = ImageFont.load_default()

        # Background for label text
        text_bbox = draw.textbbox((x1, y1 - 16), label, font=font)
        draw.rectangle(text_bbox, fill=(0, 0, 0, 180))
        draw.text(
            (x1, y1 - 16),
            label,
            fill=(255, 100, 140),
            font=font,
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    logger.info("Generated change overlay with %d region boxes", len(regions))

    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_array_to_base64(arr: np.ndarray) -> str:
    """Convert a float32 [0,1] image array to a base64 PNG."""
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
