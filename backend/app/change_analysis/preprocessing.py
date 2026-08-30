"""
Image preprocessing for bi-temporal change analysis.

Handles loading, normalisation, resizing, and optional GeoTIFF metadata
extraction.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("satquery.change.preprocessing")


@dataclass
class GeoMetadata:
    """Geo-spatial metadata extracted from a GeoTIFF header."""
    crs: Optional[str] = None
    bounds: Optional[str] = None
    width: int = 0
    height: int = 0
    bands: int = 0
    transform: Optional[str] = None


def _try_extract_geo(image_bytes: bytes) -> Optional[GeoMetadata]:
    """
    Attempt to read GeoTIFF metadata via rasterio.

    Returns None if rasterio is not installed or the file is not a GeoTIFF.
    Never fabricates data — only returns what the file header contains.
    """
    try:
        import rasterio
        with rasterio.open(io.BytesIO(image_bytes)) as src:
            crs_str = str(src.crs) if src.crs else None
            bounds_str = (
                f"({src.bounds.left:.6f}, {src.bounds.bottom:.6f}, "
                f"{src.bounds.right:.6f}, {src.bounds.top:.6f})"
            ) if src.bounds else None
            return GeoMetadata(
                crs=crs_str,
                bounds=bounds_str,
                width=src.width,
                height=src.height,
                bands=src.count,
                transform=str(src.transform) if src.transform else None,
            )
    except Exception:
        return None


def load_and_normalise(
    image_bytes: bytes,
) -> Tuple[np.ndarray, Optional[GeoMetadata]]:
    """
    Load raw image bytes into a normalised (H, W, 3) float32 RGB array
    in [0, 1] range.  Also returns any GeoTIFF metadata found.
    """
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB regardless of source mode (handles RGBA, L, P, etc.)
    image = image.convert("RGB")

    arr = np.asarray(image, dtype=np.float32) / 255.0

    geo = _try_extract_geo(image_bytes)

    logger.info(
        "Loaded image: shape=%s, geo=%s",
        arr.shape,
        "yes" if geo else "no",
    )

    return arr, geo


def make_compatible(
    before: np.ndarray,
    after: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure both images have the same spatial dimensions by resizing the
    larger one to match the smaller.

    Both inputs and outputs are (H, W, 3) float32 arrays in [0, 1].
    """
    h1, w1 = before.shape[:2]
    h2, w2 = after.shape[:2]

    if (h1, w1) == (h2, w2):
        return before, after

    # Use the smaller dimensions to avoid up-scaling artefacts.
    target_h = min(h1, h2)
    target_w = min(w1, w2)

    def _resize(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
        if arr.shape[0] == th and arr.shape[1] == tw:
            return arr
        img = Image.fromarray((arr * 255).astype(np.uint8))
        img = img.resize((tw, th), Image.LANCZOS)
        return np.asarray(img, dtype=np.float32) / 255.0

    before = _resize(before, target_h, target_w)
    after = _resize(after, target_h, target_w)

    logger.info("Aligned to common size: (%d, %d)", target_h, target_w)

    return before, after


def image_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a float32 [0,1] array back to a PIL Image."""
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def pil_to_base64(img: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG string."""
    import base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
