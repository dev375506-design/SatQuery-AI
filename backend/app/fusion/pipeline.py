"""
Optical + SAR Multimodal Fusion Pipeline.

Orchestrates the scientific end-to-end flow:
    Validation -> Modality-Specific Preprocessing -> Spatial Alignment ->
    Feature Extraction -> Multimodal Fusion -> Visualisation ->
    Query-Aware Reasoning -> Result Assembly & Detailed Diagnostics
"""

from __future__ import annotations

import io
import logging
import time
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image

from app.fusion.alignment import align_modalities
from app.fusion.feature_extraction import (
    extract_optical_features,
    extract_sar_features,
)
from app.fusion.fusion_engine import FusionEngine
from app.fusion.preprocessing import load_optical, load_sar
from app.fusion.reasoning import reason_multimodal
from app.fusion.schemas import FusionAnalysisResult
from app.fusion.visualization import (
    generate_evidence_map,
    generate_fusion_composite,
    generate_optical_preview,
    generate_sar_preview,
)

logger = logging.getLogger("satquery.fusion.pipeline")


def run_fusion_analysis(
    optical_bytes: bytes,
    sar_bytes: bytes,
    query: str = "",
    fusion_method: str = "composite",
) -> FusionAnalysisResult:
    """
    Execute the complete Optical + SAR multimodal fusion pipeline.
    """
    query_str = query or "Analyze this scene using fused optical and SAR satellite imagery."
    result = FusionAnalysisResult(query=query_str)
    trace = result.execution_trace
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Input Validation
    # ------------------------------------------------------------------
    trace.append("Input validation started")

    if not optical_bytes:
        raise ValueError("Optical image is empty or missing.")
    if not sar_bytes:
        raise ValueError("SAR image is empty or missing.")

    trace.append("✓ Both optical and SAR image payloads received")

    # ------------------------------------------------------------------
    # 2. Modality-Specific Preprocessing
    # ------------------------------------------------------------------
    trace.append("Optical preprocessing (radiometric normalization, Sentinel-2 reflectance scaling, NDWI/NDVI)")
    opt_raw, geo_opt, opt_info = load_optical(optical_bytes)
    b8_text = f", B08 NIR: {opt_info.get('b08_mean', 'N/A')}" if "b08_mean" in opt_info else ""
    trace.append(
        f"✓ Optical loaded: {opt_raw.rgb.shape[1]}x{opt_raw.rgb.shape[0]} px "
        f"({opt_info['format']}, B02: {opt_info.get('b02_mean')}, B03: {opt_info.get('b03_mean')}, B04: {opt_info.get('b04_mean')}{b8_text})"
    )

    trace.append("SAR preprocessing (Lee speckle reduction, dual-pol extraction, physical dB calibration)")
    sar_raw, geo_sar, sar_info = load_sar(sar_bytes)
    vh_text = f", VH mean: {sar_info.get('vh_mean_db')} dB" if "vh_mean_db" in sar_info else ""
    trace.append(
        f"✓ SAR loaded: {sar_raw.vv_db.shape[1]}x{sar_raw.vv_db.shape[0]} px "
        f"({sar_info['polarization']}, VV mean: {sar_info.get('vv_mean_db')} dB [{sar_info.get('vv_min_db')}, {sar_info.get('vv_max_db')} dB]{vh_text})"
    )

    # ------------------------------------------------------------------
    # 3. Spatial Alignment
    # ------------------------------------------------------------------
    trace.append("Spatial alignment (Geospatial CRS / transform intersection or dimension matching)")
    opt_aligned, sar_aligned, align_info = align_modalities(
        optical=opt_raw,
        sar=sar_raw,
        geo_opt=geo_opt,
        geo_sar=geo_sar,
    )
    result.alignment_info = align_info
    result.geo_crs = align_info.crs
    result.geo_bounds = align_info.bounds

    if align_info.method == "geo_referenced":
        trace.append(
            f"✓ Geo-referenced reprojection verified (CRS: {align_info.crs}, "
            f"grid: {align_info.target_shape[1]}x{align_info.target_shape[0]} px, res: {align_info.common_grid_resolution})"
        )
    else:
        trace.append(f"✓ {align_info.details}")

    # ------------------------------------------------------------------
    # 4. Physical Feature Extraction
    # ------------------------------------------------------------------
    trace.append("Extracting optical spectral indices (NDWI, NDVI) and surface water candidate signatures")
    opt_features = extract_optical_features(opt_aligned)
    water_opt_pct = opt_features.stats.get("optical_water_percentage", 0.0)
    trace.append(
        f"✓ Optical features extracted (NDWI thresh: {opt_features.ndwi_threshold:+.2f}, "
        f"optical water candidate: {water_opt_pct:.2f}%)"
    )

    trace.append("Extracting radar backscatter distributions, adaptive thresholds, double-bounce and specular masks")
    sar_features = extract_sar_features(sar_aligned)
    double_pct = sar_features.stats.get("double_bounce_percentage", 0.0)
    spec_pct = sar_features.stats.get("sar_water_percentage", 0.0)
    trace.append(
        f"✓ SAR features extracted (water thresh: {sar_features.water_threshold_db:.1f} dB, "
        f"{spec_pct:.2f}% specular candidate, {double_pct:.2f}% double-bounce structures)"
    )

    # ------------------------------------------------------------------
    # 5. Multimodal Fusion Engine
    # ------------------------------------------------------------------
    trace.append(f"Cross-modal fusion execution (method: {fusion_method})")
    engine = FusionEngine(method=fusion_method)
    fused_data = engine.fuse(opt_features, sar_features)
    result.features = fused_data.features
    result.modality_agreement_percentage = fused_data.overall_agreement_pct
    trace.append(
        f"✓ Multimodal fusion complete: {fused_data.overall_agreement_pct}% physical sensor agreement "
        f"(Inundation IoU: {fused_data.metrics.get('inundation_iou_pct', 0.0):.1f}%), "
        f"{len(fused_data.features)} isolated cross-modal feature regions"
    )

    # ------------------------------------------------------------------
    # 6. Visual Product Generation
    # ------------------------------------------------------------------
    trace.append("Generating optical preview, calibrated SAR dB preview, fused composite, and agreement heatmap")
    result.optical_image_b64 = generate_optical_preview(opt_features.rgb)
    result.sar_image_b64 = generate_sar_preview(sar_features.db_map_norm)

    chosen_composite = fused_data.ihs_fused_rgb if fusion_method == "ihs" else fused_data.composite_rgb
    result.fusion_visualization_b64 = generate_fusion_composite(chosen_composite, fused_data.features)
    result.evidence_map_b64 = generate_evidence_map(fused_data.agreement_heatmap)
    trace.append("✓ Visual products encoded to base64 PNG")

    # ------------------------------------------------------------------
    # 7. Query-Aware Multimodal Reasoning & Diagnostics
    # ------------------------------------------------------------------
    trace.append("Query-aware multimodal reasoning and scientific evidence synthesis")

    comp_img = Image.fromarray((np.clip(chosen_composite, 0.0, 1.0) * 255).astype("uint8"))
    comp_buf = io.BytesIO()
    comp_img.save(comp_buf, format="PNG")
    fused_image_bytes = comp_buf.getvalue()

    summary, opt_ev, sar_ev, fused_interp, conf, is_sim = reason_multimodal(
        query=query_str,
        opt=opt_features,
        sar=sar_features,
        metrics=fused_data.metrics,
        features=fused_data.features,
        fused_image_bytes=fused_image_bytes,
    )

    result.summary = summary
    result.optical_evidence = opt_ev
    result.sar_evidence = sar_ev
    result.fused_interpretation = fused_interp
    result.confidence = conf
    result.simulated = is_sim

    # Full Transparent Scientific Diagnostics Package
    result.diagnostics = {
        # Optical real measurements
        "optical_stats": opt_features.stats,
        "optical_water_candidate_pct": water_opt_pct,
        "ndwi_threshold": opt_features.ndwi_threshold,
        "optical_crs": align_info.optical_crs,
        "optical_bounds": align_info.optical_bounds,

        # SAR real measurements
        "sar_stats": sar_features.stats,
        "sar_water_candidate_pct": spec_pct,
        "sar_double_bounce_pct": double_pct,
        "water_backscatter_threshold_db": sar_features.water_threshold_db,
        "double_bounce_threshold_db": sar_features.double_bounce_threshold_db,
        "sar_crs": align_info.sar_crs,
        "sar_bounds": align_info.sar_bounds,

        # Fusion consensus real measurements
        "fusion_metrics": fused_data.metrics,
        "valid_overlap_pct": fused_data.metrics.get("valid_overlap_pct", 100.0),
        "optical_only_water_pct": fused_data.metrics.get("optical_only_water_pct", 0.0),
        "sar_only_water_pct": fused_data.metrics.get("sar_only_water_pct", 0.0),
        "optical_sar_consensus_water_pct": fused_data.metrics.get("optical_sar_consensus_water_pct", 0.0),
        "consensus_dry_land_pct": fused_data.metrics.get("consensus_dry_land_pct", 0.0),
        "modality_agreement_pct": fused_data.metrics.get("modality_agreement_pct", 0.0),
        "inundation_iou_pct": fused_data.metrics.get("inundation_iou_pct", 0.0),
        "disagreement_pct": fused_data.metrics.get("disagreement_pct", 0.0),
        "number_of_regions": len(fused_data.features),
        "total_candidate_inundation_area_pct": fused_data.metrics.get("total_candidate_inundation_area_pct", 0.0),
        "permanent_water_handling_status": fused_data.metrics.get("permanent_water_handling_status", ""),

        # Alignment diagnostics
        "alignment_method": align_info.method,
        "geo_metadata_used": align_info.geo_used,
        "common_crs": align_info.common_crs,
        "common_grid_resolution": align_info.common_grid_resolution,
        "alignment_details": align_info.details,
    }

    if is_sim:
        trace.append("✓ Multimodal interpretation complete (rule-based reasoning engine)")
    else:
        trace.append("✓ Multimodal interpretation complete (LLaVA VLM)")

    # ------------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    trace.append(f"Pipeline complete in {elapsed:.2f}s")

    logger.info(
        "Fusion pipeline finished: query=%r, conf=%d%%, agreement=%.2f%%, IoU=%.2f%%, features=%d, time=%.2fs",
        query_str, conf, result.modality_agreement_percentage, fused_data.metrics.get("inundation_iou_pct", 0.0), len(result.features), elapsed
    )

    return result
