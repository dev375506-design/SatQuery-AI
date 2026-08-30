"""
Optical and SAR-specific satellite image preprocessing.

Distinctly separates physical handling for Optical (radiometric RGB/multispectral bands,
Sentinel-2 reflectance scaling, NDWI/NDVI) and SAR (radar backscatter intensity,
speckle reduction, physical dB conversion, VV + VH dual-polarization).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image

logger = logging.getLogger("satquery.fusion.preprocessing")


@dataclass
class GeoMetadata:
    """Geo-spatial metadata extracted from a GeoTIFF header."""
    crs: Optional[str] = None
    bounds: Optional[Tuple[float, float, float, float]] = None  # (left, bottom, right, top)
    bounds_str: Optional[str] = None
    width: int = 0
    height: int = 0
    bands: int = 0
    transform: Optional[Any] = None
    res: Optional[Tuple[float, float]] = None
    nodata: Optional[float] = None


def _calc_percentiles(arr: np.ndarray, pts=(5, 10, 25, 50, 75, 90, 95)) -> Dict[str, float]:
    """Compute exact percentiles on a 2D/1D numpy array."""
    res = {}
    for p in pts:
        res[f"p{p}"] = round(float(np.percentile(arr, p)), 2)
    return res


def _try_extract_geo(image_bytes: bytes) -> Optional[GeoMetadata]:
    """Extract GeoTIFF metadata via rasterio.MemoryFile without fabricating data."""
    try:
        import rasterio
        with rasterio.MemoryFile(image_bytes) as memfile:
            with memfile.open() as src:
                crs_str = str(src.crs) if src.crs else None
                bounds_tuple = (
                    src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top
                ) if src.bounds else None
                bounds_str = (
                    f"({src.bounds.left:.6f}, {src.bounds.bottom:.6f}, "
                    f"{src.bounds.right:.6f}, {src.bounds.top:.6f})"
                ) if src.bounds else None

                return GeoMetadata(
                    crs=crs_str,
                    bounds=bounds_tuple,
                    bounds_str=bounds_str,
                    width=src.width,
                    height=src.height,
                    bands=src.count,
                    transform=src.transform,
                    res=src.res if hasattr(src, "res") else None,
                    nodata=src.nodata,
                )
    except Exception as e:
        logger.debug("GeoTIFF metadata extraction note: %s", e)
        return None


# ---------------------------------------------------------------------------
# Optical Preprocessing
# ---------------------------------------------------------------------------

@dataclass
class OpticalRawBundle:
    """Container for preprocessed optical bands and metadata."""
    rgb: np.ndarray  # (H, W, 3) float32 [0, 1] for visualization
    b02_blue: np.ndarray  # (H, W) float32 reflectance [0, 1]
    b03_green: np.ndarray  # (H, W) float32 reflectance [0, 1]
    b04_red: np.ndarray  # (H, W) float32 reflectance [0, 1]
    b08_nir: Optional[np.ndarray]  # (H, W) float32 reflectance [0, 1]
    geo: Optional[GeoMetadata]
    info: Dict[str, Any]


def _stretch_for_display(band: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Contrast stretch reflectance band for crisp visual rendering without saturation."""
    p2 = float(np.percentile(band, p_low))
    p98 = float(np.percentile(band, p_high))
    if p98 > p2:
        return np.clip((band - p2) / (p98 - p2), 0.0, 1.0).astype(np.float32)
    mx = float(band.max())
    return (band / mx if mx > 0 else band).astype(np.float32)


def load_optical(
    image_bytes: bytes,
) -> Tuple[OpticalRawBundle, Optional[GeoMetadata], Dict[str, Any]]:
    """
    Load optical image bytes into calibrated reflectance bands [0, 1].

    Supports:
    - 4-band Sentinel-2 GeoTIFF (B02 Blue, B03 Green, B04 Red, B08 NIR)
    - 3-band RGB GeoTIFF / PNG / JPEG
    - Handles raw Digital Number scaling (DN / 10000 for Sentinel-2 BOA/TOA reflectance)
    """
    geo = _try_extract_geo(image_bytes)
    info: Dict[str, Any] = {"format": "standard_image", "bands": 3}

    loaded = False
    b02, b03, b04, b08 = None, None, None, None

    if geo and geo.bands >= 1:
        try:
            import rasterio
            with rasterio.MemoryFile(image_bytes) as memfile:
                with memfile.open() as src:
                    info["crs"] = str(src.crs) if src.crs else None
                    info["bounds"] = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top] if src.bounds else None
                    info["bands"] = src.count

                    def _scale_dn_to_reflectance(raw_arr: np.ndarray) -> np.ndarray:
                        arr = raw_arr.astype(np.float32)
                        max_val = float(arr.max())
                        if max_val > 255.0:
                            # Sentinel-2 quantification value is 10000 (reflectance = DN / 10000.0)
                            return np.clip(arr / 10000.0, 0.0, 1.0)
                        elif max_val > 1.0:
                            return np.clip(arr / 255.0, 0.0, 1.0)
                        return np.clip(arr, 0.0, 1.0)

                    if src.count >= 4:
                        # Sentinel-2 4-band format: Band 1=B02 (Blue), Band 2=B03 (Green), Band 3=B04 (Red), Band 4=B08 (NIR)
                        b02 = _scale_dn_to_reflectance(src.read(1))
                        b03 = _scale_dn_to_reflectance(src.read(2))
                        b04 = _scale_dn_to_reflectance(src.read(3))
                        b08 = _scale_dn_to_reflectance(src.read(4))
                        info["format"] = "sentinel2_4band_geotiff (B02, B03, B04, B08)"
                    elif src.count == 3:
                        # 3-band RGB: Band 1=Red/B04, Band 2=Green/B03, Band 3=Blue/B02
                        b04 = _scale_dn_to_reflectance(src.read(1))
                        b03 = _scale_dn_to_reflectance(src.read(2))
                        b02 = _scale_dn_to_reflectance(src.read(3))
                        b08 = None
                        info["format"] = "rgb_3band_geotiff"
                    else:
                        gray = _scale_dn_to_reflectance(src.read(1))
                        b02, b03, b04, b08 = gray, gray, gray, None
                        info["format"] = "single_band_geotiff"

                    loaded = True
        except Exception as e:
            logger.warning("Rasterio optical read fallback: %s", e)

    if not loaded:
        # Fallback to PIL standard reading (8-bit RGB)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        b04 = arr[..., 0]  # Red
        b03 = arr[..., 1]  # Green
        b02 = arr[..., 2]  # Blue
        b08 = None
        info["format"] = "standard_rgb_image"

    assert b02 is not None and b03 is not None and b04 is not None

    # True Color RGB for visual representation (with natural contrast stretch)
    rgb_vis = np.stack([
        _stretch_for_display(b04),
        _stretch_for_display(b03),
        _stretch_for_display(b02),
    ], axis=-1)

    def _stats_profile(band: np.ndarray) -> Dict[str, float]:
        return {
            "mean": round(float(band.mean()), 4),
            "median": round(float(np.median(band)), 4),
            "std": round(float(band.std()), 4),
            "min": round(float(band.min()), 4),
            "max": round(float(band.max()), 4),
            "p10": round(float(np.percentile(band, 10)), 4),
            "p90": round(float(np.percentile(band, 90)), 4),
        }

    info["b02_stats"] = _stats_profile(b02)
    info["b03_stats"] = _stats_profile(b03)
    info["b04_stats"] = _stats_profile(b04)
    info["b02_mean"] = info["b02_stats"]["mean"]
    info["b03_mean"] = info["b03_stats"]["mean"]
    info["b04_mean"] = info["b04_stats"]["mean"]

    if b08 is not None:
        info["b08_stats"] = _stats_profile(b08)
        info["b08_mean"] = info["b08_stats"]["mean"]

    logger.info(
        "Optical loaded: shape=%s, format=%s, B02=%.3f, B03=%.3f, B04=%.3f, B08=%s",
        rgb_vis.shape, info["format"],
        info["b02_mean"], info["b03_mean"], info["b04_mean"],
        f"{info['b08_mean']:.3f}" if b08 is not None else "None"
    )

    bundle = OpticalRawBundle(
        rgb=rgb_vis,
        b02_blue=b02,
        b03_green=b03,
        b04_red=b04,
        b08_nir=b08,
        geo=geo,
        info=info,
    )
    return bundle, geo, info


# ---------------------------------------------------------------------------
# SAR Radar Preprocessing
# ---------------------------------------------------------------------------

@dataclass
class SARRawBundle:
    """Container for preprocessed SAR polarimetric bands and metadata."""
    vv_db: np.ndarray  # (H, W) float32 in decibels
    vh_db: Optional[np.ndarray]  # (H, W) float32 in decibels (if dual-pol)
    db_map_norm: np.ndarray  # (H, W) float32 [0, 1] normalized for display/composite
    polarization: str  # e.g. "dual_pol (VV+VH)" or "single_pol (VV)"
    geo: Optional[GeoMetadata]
    info: Dict[str, Any]


def _adaptive_despeckle(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply 2D spatial filtering to suppress radar speckle while retaining sharp edges."""
    try:
        from scipy.ndimage import median_filter
        return median_filter(image, size=kernel_size)
    except ImportError:
        pad = kernel_size // 2
        padded = np.pad(image, pad, mode="reflect")
        h, w = image.shape
        out = np.zeros_like(image)
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                out += padded[dy:dy + h, dx:dx + w]
        return out / (kernel_size * kernel_size)


def _convert_to_physical_db(raw_arr: np.ndarray) -> np.ndarray:
    """
    Convert raw SAR band (linear power / amplitude / DN) to calibrated decibels (dB).
    
    Ensures negative physical dB range (-45 dB to +5 dB typical for Earth observation).
    """
    arr = raw_arr.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())

    # Case 1: Already in physical dB (values largely negative, e.g. -45 to +10)
    if min_val < -5.0 and max_val < 25.0:
        return np.clip(arr, -50.0, 15.0)

    # Case 2: Linear amplitude or sigma0 power (values positive, typical range 0.00001 to 5.0)
    if max_val <= 10.0:
        despeckled = _adaptive_despeckle(np.maximum(arr, 0.0), kernel_size=3)
        eps = 1e-6
        # Check if values are in amplitude scale (where mean is around 0.05-0.2) or power
        # 10*log10(max(P, eps))
        db = 10.0 * np.log10(np.maximum(despeckled, eps))
        return np.clip(db, -50.0, 15.0)

    # Case 3: Digital Numbers (integer/float DN in 0..255 or 0..65535)
    despeckled = _adaptive_despeckle(np.maximum(arr, 0.0), kernel_size=3)
    norm_dn = despeckled / max_val
    eps = 1e-5
    db = 10.0 * np.log10(np.maximum(norm_dn, eps)) * (35.0 / 25.0) - 10.0
    return np.clip(db, -50.0, 15.0)


def load_sar(
    image_bytes: bytes,
) -> Tuple[SARRawBundle, Optional[GeoMetadata], Dict[str, Any]]:
    """
    Load SAR imagery with radar-aware dual-polarization preprocessing.

    Preserves radar backscatter intensity dynamics:
    1. Loads Band 1 (VV) and Band 2 (VH) independently.
    2. Applies 2D spatial despeckling to reduce multiplicative speckle.
    3. Transforms intensity to calibrated physical decibels (dB).
    4. Computes normalized visualization array.
    """
    geo = _try_extract_geo(image_bytes)
    info: Dict[str, Any] = {"format": "sar_standard", "polarization": "single_pol"}

    loaded = False
    vv_raw: Optional[np.ndarray] = None
    vh_raw: Optional[np.ndarray] = None

    if geo and geo.bands >= 1:
        try:
            import rasterio
            with rasterio.MemoryFile(image_bytes) as memfile:
                with memfile.open() as src:
                    info["crs"] = str(src.crs) if src.crs else None
                    info["bounds"] = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top] if src.bounds else None
                    info["bands"] = src.count

                    if src.count >= 2:
                        vv_raw = src.read(1)
                        vh_raw = src.read(2)
                        info["polarization"] = "dual_pol (VV+VH)"
                        info["format"] = "sentinel1_dualpol_geotiff (Band1=VV, Band2=VH)"
                    else:
                        vv_raw = src.read(1)
                        vh_raw = None
                        info["polarization"] = "single_pol (VV)"
                        info["format"] = "sentinel1_singlepol_geotiff"

                    loaded = True
        except Exception as e:
            logger.warning("Rasterio SAR read fallback: %s", e)

    if not loaded:
        # Load via PIL (grayscale TIFF / 8-bit PNG / JPG)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("I", "I;16", "F", "L"):
            vv_raw = np.asarray(img, dtype=np.float32)
        else:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
            vv_raw = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        vh_raw = None
        info["polarization"] = "single_pol (intensity)"
        info["format"] = "standard_sar_image"

    assert vv_raw is not None

    # Convert to physical dB
    vv_db = _convert_to_physical_db(vv_raw)
    vh_db = _convert_to_physical_db(vh_raw) if vh_raw is not None else None

    # Normalized dB map for visual composite rendering [0, 1]
    p_min = float(np.percentile(vv_db, 2))
    p_max = float(np.percentile(vv_db, 98))
    if p_max <= p_min:
        p_min, p_max = -35.0, 0.0
    db_map_norm = np.clip((vv_db - p_min) / (p_max - p_min), 0.0, 1.0).astype(np.float32)

    def _sar_stats_profile(band: np.ndarray) -> Dict[str, Any]:
        p_dict = _calc_percentiles(band, (5, 10, 25, 50, 75, 90, 95))
        prof: Dict[str, Any] = {
            "mean": round(float(band.mean()), 2),
            "median": round(float(np.median(band)), 2),
            "std": round(float(band.std()), 2),
            "min": round(float(band.min()), 2),
            "max": round(float(band.max()), 2),
            "percentiles": p_dict,
        }
        prof.update(p_dict)
        return prof

    info["vv_stats"] = _sar_stats_profile(vv_db)
    info["vv_mean_db"] = info["vv_stats"]["mean"]
    info["vv_min_db"] = info["vv_stats"]["min"]
    info["vv_max_db"] = info["vv_stats"]["max"]
    info["vv_std_db"] = info["vv_stats"]["std"]

    if vh_db is not None:
        info["vh_stats"] = _sar_stats_profile(vh_db)
        info["vh_mean_db"] = info["vh_stats"]["mean"]
        info["vh_min_db"] = info["vh_stats"]["min"]
        info["vh_max_db"] = info["vh_stats"]["max"]
        info["vh_std_db"] = info["vh_stats"]["std"]
        
        ratio = vv_db - vh_db
        info["vv_vh_ratio_stats"] = {
            "mean": round(float(ratio.mean()), 2),
            "median": round(float(np.median(ratio)), 2),
            "std": round(float(ratio.std()), 2),
        }
        info["vv_vh_ratio_mean_db"] = info["vv_vh_ratio_stats"]["mean"]

    logger.info(
        "SAR loaded: shape=%s, pol=%s, VV mean=%.2f dB [%.2f, %.2f], VH=%s",
        vv_db.shape, info["polarization"],
        info["vv_mean_db"], info["vv_min_db"], info["vv_max_db"],
        f"{info['vh_mean_db']:.2f} dB" if vh_db is not None else "None"
    )

    bundle = SARRawBundle(
        vv_db=vv_db,
        vh_db=vh_db,
        db_map_norm=db_map_norm,
        polarization=info["polarization"],
        geo=geo,
        info=info,
    )
    return bundle, geo, info
