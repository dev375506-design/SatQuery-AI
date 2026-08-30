"""
Physical feature extraction for Optical and SAR satellite imagery.

Extracts physically grounded Earth observation indicators:
- Optical: Calibrated NDWI, NDVI, NIR absorption tests, multi-band percentiles,
  and optical water candidate masks.
- SAR: Adaptive radar backscatter decibel (dB) distributions, dual-pol VV/VH
  thresholding, specular water candidate masks, and double-bounce urban masks.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

import numpy as np

from app.fusion.preprocessing import OpticalRawBundle, SARRawBundle
from app.fusion.schemas import OpticalFeatures, SARFeatures

logger = logging.getLogger("satquery.fusion.feature_extraction")


def _calc_stats(arr: np.ndarray) -> Dict[str, float]:
    """Calculate basic distribution metrics."""
    return {
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
    }


def _compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute spatial gradient magnitude using Sobel-like finite differences."""
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)

    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0

    mag = np.sqrt(gx ** 2 + gy ** 2)
    mx = float(mag.max())
    return (mag / mx if mx > 0 else mag).astype(np.float32)


def _compute_otsu_threshold(arr: np.ndarray, min_val: float, max_val: float, bins: int = 128) -> float:
    """Compute Otsu threshold on a continuous numpy array."""
    clipped = np.clip(arr, min_val, max_val)
    hist, bin_edges = np.histogram(clipped, bins=bins, range=(min_val, max_val))
    prob = hist.astype(np.float32) / float(arr.size)

    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * (bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2.0))
    global_mean = cum_mean[-1]

    between_var = np.zeros(bins)
    for i in range(bins):
        w0 = cum_prob[i]
        w1 = 1.0 - w0
        if w0 <= 1e-6 or w1 <= 1e-6:
            continue
        u0 = cum_mean[i] / w0
        u1 = (global_mean - cum_mean[i]) / w1
        between_var[i] = w0 * w1 * ((u0 - u1) ** 2)

    best_idx = int(np.argmax(between_var))
    return float(bin_edges[best_idx])


def extract_optical_features(optical: OpticalRawBundle) -> OpticalFeatures:
    """
    Extract radiometric indices and surface classification features from calibrated optical bands.
    """
    b02 = optical.b02_blue
    b03 = optical.b03_green
    b04 = optical.b04_red
    b08 = optical.b08_nir

    # Luminance (ITU-R BT.601)
    luminance = np.clip(0.299 * b04 + 0.587 * b03 + 0.114 * b02, 0.0, 1.0)
    edges = _compute_gradient_magnitude(luminance)

    eps = 1e-6

    # 1. NDWI (Normalized Difference Water Index) & NDVI
    if b08 is not None:
        # Sentinel-2 McFeeters NDWI = (B03 - B08) / (B03 + B08)
        ndwi = (b03 - b08) / (b03 + b08 + eps)
        # Sentinel-2 NDVI = (B08 - B04) / (B08 + B04)
        ndvi = (b08 - b04) / (b08 + b04 + eps)

        ndwi_med = float(np.median(ndwi))
        # Adaptive NDWI threshold: baseline 0.05, but if scene median is high due to sediment/salinity, adjust
        ndwi_thresh = round(max(0.05, min(0.30, ndwi_med + 0.05)), 3)

        # Multi-condition optical water detection:
        # 1. Positive NDWI above threshold
        # 2. Strong NIR absorption (B08 <= 0.28 to prevent bright arid sand false positives)
        # 3. Green dominance over red (B03 >= B04 * 0.70)
        water_mask = (ndwi >= ndwi_thresh) & (b08 <= 0.28) & (b03 >= b04 * 0.70)
        veg_mask = (ndvi >= 0.30) & (b08 >= 0.20)
    else:
        # 3-band RGB proxy
        water_diff = (b02 + b03) - (2.0 * b04)
        water_denom = (b02 + b03) + (2.0 * b04) + eps
        ndwi = water_diff / water_denom
        ndvi = (2.0 * b03 - b04 - b02) / (2.0 * b03 + b04 + b02 + eps)
        ndwi_thresh = 0.15
        water_mask = (ndwi >= 0.15) & (luminance <= 0.35)
        veg_mask = (ndvi >= 0.20) & (b03 >= b04)

    mean_brightness = float(luminance.mean())

    ndwi_stats = {
        "mean": round(float(ndwi.mean()), 4),
        "median": round(float(np.median(ndwi)), 4),
        "std": round(float(ndwi.std()), 4),
        "min": round(float(ndwi.min()), 4),
        "max": round(float(ndwi.max()), 4),
        "p10": round(float(np.percentile(ndwi, 10)), 4),
        "p25": round(float(np.percentile(ndwi, 25)), 4),
        "p50": round(float(np.percentile(ndwi, 50)), 4),
        "p75": round(float(np.percentile(ndwi, 75)), 4),
        "p90": round(float(np.percentile(ndwi, 90)), 4),
    }

    stats: Dict[str, Any] = {
        "b02_stats": _calc_stats(b02),
        "b03_stats": _calc_stats(b03),
        "b04_stats": _calc_stats(b04),
        "b02_blue_mean": round(float(b02.mean()), 4),
        "b03_green_mean": round(float(b03.mean()), 4),
        "b04_red_mean": round(float(b04.mean()), 4),
        "ndwi_stats": ndwi_stats,
        "ndwi_mean": ndwi_stats["mean"],
        "ndwi_median": ndwi_stats["median"],
        "ndwi_std": ndwi_stats["std"],
        "ndwi_threshold": ndwi_thresh,
        "optical_water_percentage": round(float(water_mask.sum() / water_mask.size) * 100.0, 2),
        "water_pixel_fraction": round(float(water_mask.sum() / water_mask.size), 4),
    }

    if b08 is not None:
        stats["b08_stats"] = _calc_stats(b08)
        stats["b08_nir_mean"] = round(float(b08.mean()), 4)
        stats["ndvi_stats"] = _calc_stats(ndvi)
        stats["ndvi_mean"] = stats["ndvi_stats"]["mean"]
        stats["vegetation_pixel_fraction"] = round(float(veg_mask.sum() / veg_mask.size), 4),
        stats["vegetation_percentage"] = round(float(veg_mask.sum() / veg_mask.size) * 100.0, 2)

    logger.info(
        "Extracted Optical features: B02=%.3f, B03=%.3f, B04=%.3f, NDWI_thresh=%.3f, Water=%.1f%%",
        stats["b02_blue_mean"], stats["b03_green_mean"], stats["b04_red_mean"],
        ndwi_thresh, stats["optical_water_percentage"]
    )

    return OpticalFeatures(
        rgb=optical.rgb,
        b02_blue=b02,
        b03_green=b03,
        b04_red=b04,
        b08_nir=b08,
        ndwi=ndwi,
        ndvi=ndvi,
        luminance=luminance,
        edges=edges,
        water_mask=water_mask,
        vegetation_mask=veg_mask,
        mean_brightness=mean_brightness,
        ndwi_threshold=ndwi_thresh,
        bands_count=4 if b08 is not None else 3,
        stats=stats,
    )


def extract_sar_features(sar: SARRawBundle) -> SARFeatures:
    """
    Extract radar backscatter, adaptive physical thresholds, and scattering masks from SAR imagery.
    """
    vv_db = sar.vv_db
    vh_db = sar.vh_db

    mean_val = float(vv_db.mean())
    std_val = float(vv_db.std())
    min_val = float(vv_db.min())
    max_val = float(vv_db.max())
    med_val = float(np.median(vv_db))

    p5_vv = float(np.percentile(vv_db, 5))
    p25_vv = float(np.percentile(vv_db, 25))
    p75_vv = float(np.percentile(vv_db, 75))
    p95_vv = float(np.percentile(vv_db, 95))

    # -------------------------------------------------------------
    # 1. Adaptive SAR Water Candidate Thresholding
    # -------------------------------------------------------------
    # Calculate scene-adaptive low-backscatter water threshold
    # Grounded in radar physics: water backscatter reflects away specularly.
    # We find the lower distribution dip / Otsu on the lower half of backscatter.
    if med_val <= -22.0:
        # Acquisition is calibrated lower overall: adaptive threshold below the 35th percentile
        p35_vv = float(np.percentile(vv_db, 35))
        water_thresh_vv = round(min(-16.0, max(-38.0, p35_vv - 0.2 * std_val)), 2)
    else:
        # Standard calibration: physical threshold -14.0 dB or Otsu
        otsu_val = _compute_otsu_threshold(vv_db, min_val=-35.0, max_val=-5.0)
        water_thresh_vv = round(min(-14.0, max(-26.0, otsu_val)), 2)

    # 2. Specular Water Candidate Mask (Joint VV + VH if available)
    if vh_db is not None:
        med_vh = float(np.median(vh_db))
        std_vh = float(vh_db.std())
        p35_vh = float(np.percentile(vh_db, 35))
        water_thresh_vh = round(min(-20.0, max(-45.0, p35_vh - 0.2 * std_vh)), 2)
        specular_mask = (vv_db <= water_thresh_vv) & (vh_db <= water_thresh_vh)
        vv_vh_ratio = vv_db - vh_db
        volume_scatter_mask = vh_db >= (med_vh + 0.5 * std_vh)
    else:
        water_thresh_vh = None
        specular_mask = (vv_db <= water_thresh_vv)
        vv_vh_ratio = None
        volume_scatter_mask = None

    # 3. Double-bounce backscatter (urban walls / buildings / dihedral metal returns)
    double_bounce_thresh = round(max(-8.0, med_val + 1.5 * std_val), 2)
    double_bounce_mask = vv_db >= double_bounce_thresh

    vv_stats = sar.info.get("vv_stats", {})
    vv_stats.update({
        "mean": round(mean_val, 2),
        "median": round(med_val, 2),
        "std": round(std_val, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "p5": p5_vv,
        "p25": p25_vv,
        "p75": p75_vv,
        "p95": p95_vv,
    })

    stats: Dict[str, Any] = {
        "vv_stats": vv_stats,
        "vv_mean_db": round(mean_val, 2),
        "vv_median_db": round(med_val, 2),
        "vv_min_db": round(min_val, 2),
        "vv_max_db": round(max_val, 2),
        "vv_std_db": round(std_val, 2),
        "water_backscatter_threshold_db": water_thresh_vv,
        "double_bounce_threshold_db": double_bounce_thresh,
        "sar_water_percentage": round(float(specular_mask.sum() / specular_mask.size) * 100.0, 2),
        "specular_pixel_fraction": round(float(specular_mask.sum() / specular_mask.size), 4),
        "double_bounce_pixel_fraction": round(float(double_bounce_mask.sum() / double_bounce_mask.size), 4),
        "double_bounce_percentage": round(float(double_bounce_mask.sum() / double_bounce_mask.size) * 100.0, 2),
    }

    if vh_db is not None:
        vh_stats = sar.info.get("vh_stats", {})
        stats["vh_stats"] = vh_stats
        stats["vh_mean_db"] = vh_stats.get("mean", round(float(vh_db.mean()), 2))
        stats["vh_median_db"] = vh_stats.get("median", round(float(np.median(vh_db)), 2))
        stats["vh_min_db"] = vh_stats.get("min", round(float(vh_db.min()), 2))
        stats["vh_max_db"] = vh_stats.get("max", round(float(vh_db.max()), 2))
        stats["water_threshold_vh_db"] = water_thresh_vh
        stats["vv_vh_ratio_stats"] = sar.info.get("vv_vh_ratio_stats", {})

    logger.info(
        "Extracted SAR features: VV_mean=%.2f dB, WaterThresh=%.2f dB, SAR Water=%.1f%%, DoubleBounce=%.1f%%",
        mean_val, water_thresh_vv, stats["sar_water_percentage"],
        stats["double_bounce_percentage"]
    )

    return SARFeatures(
        vv_db=vv_db,
        vh_db=vh_db,
        vv_vh_ratio_db=vv_vh_ratio,
        db_map_norm=sar.db_map_norm,
        double_bounce_mask=double_bounce_mask,
        specular_mask=specular_mask,
        volume_scatter_mask=volume_scatter_mask,
        water_threshold_db=water_thresh_vv,
        double_bounce_threshold_db=double_bounce_thresh,
        mean_db=mean_val,
        min_db=min_val,
        max_db=max_val,
        std_db=std_val,
        polarization=sar.polarization,
        stats=stats,
    )
