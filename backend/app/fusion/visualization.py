"""
Visualisation generation for Optical + SAR Multimodal Fusion.

Generates base64-encoded PNG visual products:
1. Optical Preview
2. SAR Radar Log-Backscatter View
3. False-Color / IHS Fused Composite with labeled feature bounding boxes
4. Cross-Modal Agreement Heatmap
"""

from __future__ import annotations

import io
import base64
import logging
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.fusion.schemas import FusionFeature

logger = logging.getLogger("satquery.fusion.visualization")


def _array_to_b64(arr: np.ndarray, mode: str = "RGB") -> str:
    """Convert float32 [0, 1] array to base64 PNG."""
    arr_clipped = np.clip(arr, 0.0, 1.0)
    if arr_clipped.ndim == 2:
        img = Image.fromarray((arr_clipped * 255).astype(np.uint8), mode="L").convert("RGB")
    else:
        img = Image.fromarray((arr_clipped * 255).astype(np.uint8), mode=mode)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_optical_preview(optical: np.ndarray) -> str:
    """Generate base64 PNG preview for optical modality."""
    return _array_to_b64(optical)


def generate_sar_preview(sar_norm: np.ndarray) -> str:
    """Generate base64 PNG preview for log-scaled radar backscatter."""
    return _array_to_b64(sar_norm)


def generate_fusion_composite(
    fused_rgb: np.ndarray,
    features: List[FusionFeature],
) -> str:
    """
    Generate the fused composite image with labeled cross-modal feature bounding boxes.
    """
    img = Image.fromarray((np.clip(fused_rgb, 0.0, 1.0) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for feat in features:
        bb = feat.bbox
        x1, y1 = bb.x, bb.y
        x2, y2 = bb.x + bb.width, bb.y + bb.height

        # Determine theme color
        if "Water" in feat.category:
            box_color = (63, 216, 255)  # Cyan
            label_text = f"F{feat.feature_id}: Water ({int(feat.agreement_score * 100)}%)"
        else:
            box_color = (255, 106, 106)  # Red / Magenta
            label_text = f"F{feat.feature_id}: Structure ({int(feat.agreement_score * 100)}%)"

        # Draw bounding rectangle
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

        # Draw label badge
        text_bbox = draw.textbbox((x1, max(0, y1 - 15)), label_text, font=font)
        draw.rectangle(text_bbox, fill=(6, 10, 18, 200))
        draw.text((x1, max(0, y1 - 15)), label_text, fill=box_color, font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_evidence_map(agreement_heatmap: np.ndarray) -> str:
    """Generate base64 PNG for the quantitative cross-modal agreement map."""
    return _array_to_b64(agreement_heatmap)
