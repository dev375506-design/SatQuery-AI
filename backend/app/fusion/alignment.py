"""
Geospatial alignment engine for Optical and SAR satellite imagery.

Performs true geographic coordinate reprojection, common grid intersection,
and resampling via rasterio.warp when geospatial CRS/transform metadata is present.
Falls back to robust dimension-based matching when spatial headers are absent.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image

from app.fusion.preprocessing import GeoMetadata, OpticalRawBundle, SARRawBundle
from app.fusion.schemas import AlignmentInfo

logger = logging.getLogger("satquery.fusion.alignment")


def _reproject_band_to_grid(
    src_data: np.ndarray,
    src_crs: Any,
    src_transform: Any,
    dst_crs: Any,
    dst_transform: Any,
    dst_shape: Tuple[int, int],
    resampling_mode: str = "bilinear",
) -> np.ndarray:
    """Reproject a 2D numpy array onto a target CRS and transform grid."""
    import rasterio.warp
    from rasterio.enums import Resampling

    mode = Resampling.bilinear if resampling_mode == "bilinear" else Resampling.nearest
    dst_arr = np.zeros(dst_shape, dtype=np.float32)

    rasterio.warp.reproject(
        source=src_data.astype(np.float32),
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=mode,
    )
    return dst_arr


def _resize_2d(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Resize a 2D float32 array using Lanczos interpolation."""
    if arr.shape[0] == th and arr.shape[1] == tw:
        return arr
    min_val, max_val = float(arr.min()), float(arr.max())
    span = max_val - min_val if max_val > min_val else 1.0
    norm = ((arr - min_val) / span * 255.0).astype(np.uint8)
    img = Image.fromarray(norm, mode="L").resize((tw, th), Image.LANCZOS)
    res = np.asarray(img, dtype=np.float32) / 255.0 * span + min_val
    return res.astype(np.float32)


def align_modalities(
    optical: OpticalRawBundle,
    sar: SARRawBundle,
    geo_opt: Optional[GeoMetadata] = None,
    geo_sar: Optional[GeoMetadata] = None,
) -> Tuple[OpticalRawBundle, SARRawBundle, AlignmentInfo]:
    """
    Spatially align Optical and SAR datasets onto a common pixel grid.

    If both datasets possess valid GeoTIFF CRS and affine transforms,
    computes the geographic intersection bounding box and reprojects both
    optical and SAR bands onto a common geographic grid.

    Otherwise, performs high-quality dimension matching.
    """
    h_opt, w_opt = optical.b02_blue.shape
    h_sar, w_sar = sar.vv_db.shape

    opt_crs_str = geo_opt.crs if (geo_opt and geo_opt.crs) else None
    sar_crs_str = geo_sar.crs if (geo_sar and geo_sar.crs) else None
    opt_bounds_str = geo_opt.bounds_str if (geo_opt and geo_opt.bounds_str) else None
    sar_bounds_str = geo_sar.bounds_str if (geo_sar and geo_sar.bounds_str) else None

    has_geo = (
        geo_opt is not None
        and geo_sar is not None
        and geo_opt.crs is not None
        and geo_sar.crs is not None
        and geo_opt.transform is not None
        and geo_sar.transform is not None
        and geo_opt.bounds is not None
        and geo_sar.bounds is not None
    )

    if has_geo:
        try:
            import rasterio.warp
            from rasterio.crs import CRS
            from affine import Affine

            crs_opt = CRS.from_user_input(geo_opt.crs)
            crs_sar = CRS.from_user_input(geo_sar.crs)

            # Transform SAR bounds to Optical CRS coordinates
            sar_left, sar_bottom, sar_right, sar_top = geo_sar.bounds
            sar_b_opt = rasterio.warp.transform_bounds(
                crs_sar, crs_opt, sar_left, sar_bottom, sar_right, sar_top
            )

            opt_left, opt_bottom, opt_right, opt_top = geo_opt.bounds

            # Geographic Intersection Bounding Box (in Optical CRS units)
            inter_left = max(opt_left, sar_b_opt[0])
            inter_bottom = max(opt_bottom, sar_b_opt[1])
            inter_right = min(opt_right, sar_b_opt[2])
            inter_top = min(opt_top, sar_b_opt[3])

            # Check if valid non-empty geographic intersection exists
            if inter_right > inter_left and inter_top > inter_bottom:
                # Pixel resolution from optical affine transform
                dx = abs(geo_opt.transform.a)
                dy = abs(geo_opt.transform.e)
                if dx == 0:
                    dx = abs(sar_left - sar_right) / max(1, w_sar)
                if dy == 0:
                    dy = abs(sar_top - sar_bottom) / max(1, h_sar)

                out_w = max(5, int(round((inter_right - inter_left) / dx)))
                out_h = max(5, int(round((inter_top - inter_bottom) / dy)))

                if out_w >= 5 and out_h >= 5:
                    dst_transform = Affine.translation(inter_left, inter_top) * Affine.scale(dx, -dy)
                    dst_shape = (out_h, out_w)

                    is_degree = crs_opt.is_geographic
                    res_unit = "deg" if is_degree else "m"
                    grid_res_str = f"{dx:.6f}{res_unit} x {dy:.6f}{res_unit}"

                    logger.info(
                        "Geospatial intersection found: %dx%d pixels at resolution (%s) in CRS %s",
                        out_w, out_h, grid_res_str, crs_opt.to_string()
                    )

                    # Reproject all optical bands onto common grid
                    b02_aligned = _reproject_band_to_grid(
                        optical.b02_blue, crs_opt, geo_opt.transform, crs_opt, dst_transform, dst_shape
                    )
                    b03_aligned = _reproject_band_to_grid(
                        optical.b03_green, crs_opt, geo_opt.transform, crs_opt, dst_transform, dst_shape
                    )
                    b04_aligned = _reproject_band_to_grid(
                        optical.b04_red, crs_opt, geo_opt.transform, crs_opt, dst_transform, dst_shape
                    )
                    b08_aligned = _reproject_band_to_grid(
                        optical.b08_nir, crs_opt, geo_opt.transform, crs_opt, dst_transform, dst_shape
                    ) if optical.b08_nir is not None else None

                    # Reproject all SAR bands onto common grid
                    vv_aligned = _reproject_band_to_grid(
                        sar.vv_db, crs_sar, geo_sar.transform, crs_opt, dst_transform, dst_shape
                    )
                    vh_aligned = _reproject_band_to_grid(
                        sar.vh_db, crs_sar, geo_sar.transform, crs_opt, dst_transform, dst_shape
                    ) if sar.vh_db is not None else None

                    # Generate aligned visual products
                    p_min = float(np.percentile(vv_aligned, 2))
                    p_max = float(np.percentile(vv_aligned, 98))
                    if p_max <= p_min:
                        p_min, p_max = -35.0, 0.0
                    sar_norm_aligned = np.clip((vv_aligned - p_min) / (p_max - p_min), 0.0, 1.0).astype(np.float32)

                    def _stretch(b):
                        p2, p98 = np.percentile(b, 2), np.percentile(b, 98)
                        return np.clip((b - p2) / (p98 - p2), 0.0, 1.0).astype(np.float32) if p98 > p2 else b

                    rgb_aligned = np.stack([_stretch(b04_aligned), _stretch(b03_aligned), _stretch(b02_aligned)], axis=-1)

                    bounds_str = f"({inter_left:.6f}, {inter_bottom:.6f}, {inter_right:.6f}, {inter_top:.6f})"
                    details_str = (
                        f"Geospatial reprojection onto common geographic grid (CRS: {crs_opt.to_string()}, "
                        f"grid: {out_w}x{out_h} px, pixel res: {grid_res_str}, bounds: {bounds_str})"
                    )

                    info = AlignmentInfo(
                        method="geo_referenced",
                        details=details_str,
                        target_shape=dst_shape,
                        crs=crs_opt.to_string(),
                        bounds=bounds_str,
                        geo_used=True,
                        resolution_m=(dx, dy),
                        optical_crs=opt_crs_str,
                        sar_crs=sar_crs_str,
                        optical_bounds=opt_bounds_str,
                        sar_bounds=sar_bounds_str,
                        common_crs=crs_opt.to_string(),
                        common_grid_resolution=grid_res_str,
                    )

                    aligned_opt = OpticalRawBundle(
                        rgb=rgb_aligned,
                        b02_blue=b02_aligned,
                        b03_green=b03_aligned,
                        b04_red=b04_aligned,
                        b08_nir=b08_aligned,
                        geo=geo_opt,
                        info=optical.info,
                    )

                    aligned_sar = SARRawBundle(
                        vv_db=vv_aligned,
                        vh_db=vh_aligned,
                        db_map_norm=sar_norm_aligned,
                        polarization=sar.polarization,
                        geo=geo_sar,
                        info=sar.info,
                    )

                    return aligned_opt, aligned_sar, info

        except Exception as e:
            logger.warning("Geospatial reprojection encountered error; falling back to dimension matching: %s", e)

    # -------------------------------------------------------------
    # Dimension Matching Fallback (for non-geotiff PNG/JPG or non-overlapping bounds)
    # -------------------------------------------------------------
    target_h = min(h_opt, h_sar)
    target_w = min(w_opt, w_sar)

    b02_aligned = _resize_2d(optical.b02_blue, target_h, target_w)
    b03_aligned = _resize_2d(optical.b03_green, target_h, target_w)
    b04_aligned = _resize_2d(optical.b04_red, target_h, target_w)
    b08_aligned = _resize_2d(optical.b08_nir, target_h, target_w) if optical.b08_nir is not None else None

    vv_aligned = _resize_2d(sar.vv_db, target_h, target_w)
    vh_aligned = _resize_2d(sar.vh_db, target_h, target_w) if sar.vh_db is not None else None

    p_min = float(np.percentile(vv_aligned, 2))
    p_max = float(np.percentile(vv_aligned, 98))
    if p_max <= p_min:
        p_min, p_max = -35.0, 0.0
    sar_norm_aligned = np.clip((vv_aligned - p_min) / (p_max - p_min), 0.0, 1.0).astype(np.float32)

    def _stretch(b):
        p2, p98 = np.percentile(b, 2), np.percentile(b, 98)
        return np.clip((b - p2) / (p98 - p2), 0.0, 1.0).astype(np.float32) if p98 > p2 else b

    rgb_aligned = np.stack([_stretch(b04_aligned), _stretch(b03_aligned), _stretch(b02_aligned)], axis=-1)

    if h_opt == h_sar and w_opt == w_sar:
        details_str = "Spatial metadata unavailable or non-intersecting; dimensions identical (1:1 pixel matching)"
    else:
        details_str = (
            f"Spatial metadata unavailable or non-intersecting; dimension-based alignment performed "
            f"({w_opt}x{h_opt} opt, {w_sar}x{h_sar} sar -> {target_w}x{target_h} common grid)"
        )

    info = AlignmentInfo(
        method="dimension_matched",
        details=details_str,
        target_shape=(target_h, target_w),
        crs=opt_crs_str or sar_crs_str,
        bounds=opt_bounds_str or sar_bounds_str,
        geo_used=False,
        optical_crs=opt_crs_str,
        sar_crs=sar_crs_str,
        optical_bounds=opt_bounds_str,
        sar_bounds=sar_bounds_str,
        common_crs=None,
        common_grid_resolution=f"{target_w}x{target_h} px",
    )

    aligned_opt = OpticalRawBundle(
        rgb=rgb_aligned,
        b02_blue=b02_aligned,
        b03_green=b03_aligned,
        b04_red=b04_aligned,
        b08_nir=b08_aligned,
        geo=geo_opt,
        info=optical.info,
    )

    aligned_sar = SARRawBundle(
        vv_db=vv_aligned,
        vh_db=vh_aligned,
        db_map_norm=sar_norm_aligned,
        polarization=sar.polarization,
        geo=geo_sar,
        info=sar.info,
    )

    return aligned_opt, aligned_sar, info
