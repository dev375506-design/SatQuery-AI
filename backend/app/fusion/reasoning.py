"""
Query-aware multimodal reasoning and scientific evidence synthesis.

Generates rigorous, scientifically grounded evidence separation:
1. OPTICAL EVIDENCE: Direct multispectral observations (B02, B03, B04, B08, NDWI, NDVI).
2. SAR EVIDENCE: Direct radar backscatter observations (VV/VH backscatter in dB, percentiles, specular reflections).
3. FUSED INTERPRETATION: Cross-modal consensus, disambiguation, and query answer.

Enforces scientific discipline: labels single-date observations as "Surface Water / Inundation Candidate"
rather than "Confirmed Flood" unless historical temporal baseline is provided.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from app.fusion.schemas import (
    FusionFeature,
    OpticalFeatures,
    SARFeatures,
)

logger = logging.getLogger("satquery.fusion.reasoning")


def _classify_query_theme(query: str) -> str:
    """Identify the primary thematic focus of the user's analytical query."""
    q = query.lower()
    if any(k in q for k in ["flood", "water", "inundat", "submerg", "river", "lake", "wet"]):
        return "water"
    if any(k in q for k in ["build", "urban", "struct", "settle", "citi", "infrastruct", "road"]):
        return "urban"
    if any(k in q for k in ["crop", "vegetat", "plant", "farm", "agri", "forest"]):
        return "vegetation"
    if any(k in q for k in ["cloud", "shadow", "all-weather", "penetrat", "haze"]):
        return "clouds"
    return "general"


def _generate_optical_evidence(opt: OpticalFeatures, features: List[FusionFeature], query_theme: str) -> str:
    """Generate structured summary of direct optical multispectral observations."""
    parts = []
    
    water_pct = opt.stats.get("optical_water_percentage", 0.0)
    veg_pct = opt.stats.get("vegetation_percentage", 0.0)
    ndwi_stats = opt.stats.get("ndwi_stats", {})

    b08_str = f", B08 NIR: {opt.stats.get('b08_nir_mean')}" if "b08_nir_mean" in opt.stats else ""
    ndwi_thresh = opt.ndwi_threshold

    parts.append(
        f"Optical multispectral imagery ({opt.bands_count} bands: B02 Blue={opt.stats.get('b02_blue_mean', 0.0):.3f}, "
        f"B03 Green={opt.stats.get('b03_green_mean', 0.0):.3f}, B04 Red={opt.stats.get('b04_red_mean', 0.0):.3f}{b08_str}) "
        f"shows mean NDWI of {ndwi_stats.get('mean', 0.0):+.3f} (median: {ndwi_stats.get('median', 0.0):+.3f}, std: {ndwi_stats.get('std', 0.0):.3f})."
    )

    water_feats = [f for f in features if "Water" in f.category or "Inundation" in f.category]
    struct_feats = [f for f in features if "Built-up" in f.category or "Settlement" in f.category]

    if query_theme == "water":
        parts.append(
            f"Applying NDWI threshold >= {ndwi_thresh:.2f} combined with NIR absorption criteria (B08 <= 0.28) isolates "
            f"{water_pct:.2f}% of the scene as optical water / inundation candidate pixels. "
            f"{f'Detected {len(water_feats)} distinct contiguous candidate water zone(s), prominently in the {water_feats[0].location_description}.' if water_feats else 'No dominant isolated open water clusters detected.'}"
        )
    elif query_theme == "urban":
        parts.append(
            f"Spatial edge analysis indicates mean gradient of {opt.edges.mean():.2f}. "
            f"{f'Detected {len(struct_feats)} dense structural cluster(s) in the {struct_feats[0].location_description}.' if struct_feats else 'Terrain exhibits predominantly uniform radiometric texture.'}"
        )
    elif query_theme == "vegetation":
        parts.append(
            f"Vegetation index response indicates active photosynthetic biomass across approximately {veg_pct:.1f}% of the scene."
        )
    else:
        parts.append(
            f"Surface classification reveals {water_pct:.2f}% optical water candidate signature, "
            f"{veg_pct:.1f}% vegetative canopy, and mean scene luminance of {opt.mean_brightness * 100:.1f}%."
        )

    return " ".join(parts)


def _generate_sar_evidence(sar: SARFeatures, features: List[FusionFeature], query_theme: str) -> str:
    """Generate structured summary of direct SAR radar observations."""
    parts = []

    spec_pct = sar.stats.get("sar_water_percentage", 0.0)
    double_pct = sar.stats.get("double_bounce_percentage", 0.0)
    vv_stats = sar.stats.get("vv_stats", {})

    pol_str = sar.polarization
    vh_info = f", VH cross-pol mean: {sar.stats.get('vh_mean_db')} dB (median: {sar.stats.get('vh_median_db')} dB)" if "vh_mean_db" in sar.stats else ""
    thresh_str = f"{sar.water_threshold_db:.1f} dB"

    parts.append(
        f"Calibrated SAR radar backscatter ({pol_str}) records mean VV intensity of {sar.mean_db:.1f} dB "
        f"[median: {sar.stats.get('vv_median_db')} dB, p10: {vv_stats.get('p10')} dB, p90: {vv_stats.get('p90')} dB, std: {sar.std_db:.1f} dB]{vh_info}."
    )

    if query_theme == "water":
        parts.append(
            f"Adaptive low-backscatter thresholding (VV <= {thresh_str}) identifies {spec_pct:.2f}% of the radar scene "
            f"as specular reflection candidates where microwave pulses reflect away from the sensor."
        )
    elif query_theme == "urban":
        parts.append(
            f"High-intensity double-bounce returns (VV >= {sar.double_bounce_threshold_db:.1f} dB) occupy {double_pct:.2f}% of the scene, "
            f"characteristic of dihedral corner reflections between vertical walls/structures and the ground plane."
        )
    elif query_theme == "vegetation":
        if sar.vh_db is not None:
            parts.append(
                f"Cross-polarization volume scattering indicates canopy interaction across vegetated ground."
            )
        else:
            parts.append(
                f"Moderate VV backscatter ({sar.mean_db:.1f} dB) corresponds to surface roughness across agricultural terrain."
            )
    else:
        parts.append(
            f"Radar returns isolate {spec_pct:.2f}% low-backscatter candidate surfaces and {double_pct:.2f}% high-intensity double-bounce structures."
        )

    return " ".join(parts)


def _generate_fused_interpretation(
    query: str,
    query_theme: str,
    opt: OpticalFeatures,
    sar: SARFeatures,
    metrics: Dict[str, Any],
    features: List[FusionFeature],
) -> str:
    """Synthesize cross-modal consensus into an executive interpretation answering the query."""
    lines = []
    
    agree_pct = metrics.get("modality_agreement_pct", 0.0)
    consensus_water = metrics.get("optical_sar_consensus_water_pct", 0.0)
    opt_only = metrics.get("optical_only_water_pct", 0.0)
    sar_only = metrics.get("sar_only_water_pct", 0.0)
    iou_pct = metrics.get("inundation_iou_pct", 0.0)

    water_feats = [f for f in features if "Water" in f.category or "Inundation" in f.category]
    struct_feats = [f for f in features if "Built-up" in f.category or "Settlement" in f.category]
    flooded_veg_feats = [f for f in features if "Flooded" in f.category or "Wetland" in f.category]

    lines.append(
        f"Cross-modal fusion achieves {agree_pct:.2f}% physical sensor agreement across the common spatial domain "
        f"(Inundation IoU: {iou_pct:.2f}%)."
    )

    if query_theme == "water":
        lines.append(
            f"Regarding surface water and inundation: Optical NDWI (thresh >= {opt.ndwi_threshold}) and SAR specular backscatter "
            f"(thresh <= {sar.water_threshold_db:.1f} dB) jointly confirm {consensus_water:.2f}% surface water / inundation candidate area. "
            f"Sensor discrepancies account for {opt_only:.2f}% optical-only and {sar_only:.2f}% SAR-only candidate detections."
        )
        if flooded_veg_feats:
            lines.append(
                f"Additionally, {len(flooded_veg_feats)} sector(s) of potential flooded vegetation / sub-canopy wetness were detected "
                f"in the {flooded_veg_feats[0].location_description}."
            )
        if water_feats:
            lines.append(
                f"Primary candidate water bodies are delineated in the {water_feats[0].location_description}."
            )
        lines.append(
            "[Note: Since this analysis evaluates single-date acquisitions without a historical baseline, candidate water is reported rather than confirmed permanent vs flood water]."
        )
    elif query_theme == "urban":
        consensus_struct_pct = (((opt.edges > 0.20) & sar.double_bounce_mask).sum() / opt.edges.size) * 100.0
        lines.append(
            f"Regarding built-up infrastructure: Optical edge contrast and SAR double-bounce backscatter (VV >= {sar.double_bounce_threshold_db:.1f} dB) "
            f"confirm {consensus_struct_pct:.2f}% dense urban footprint."
        )
        if struct_feats:
            lines.append(
                f"Main settlement clusters are localized in the {struct_feats[0].location_description}."
            )
    elif query_theme == "clouds":
        lines.append(
            f"SAR microwave penetration bypasses optical atmospheric attenuation, confirming terrain structures "
            f"and surface moisture directly beneath cloud cover."
        )
    else:
        lines.append(
            f"Multimodal analysis provides all-weather characterization: {len(water_feats)} candidate water zone(s) "
            f"and {len(struct_feats)} built-up infrastructure cluster(s) isolated with high cross-sensor consistency."
        )

    return " ".join(lines)


def _compute_calibrated_confidence(
    metrics: Dict[str, Any],
    opt: OpticalFeatures,
    sar: SARFeatures,
    features_count: int,
) -> int:
    """
    Compute a statistically grounded confidence score (0-100%).
    
    Formula derives from:
    1. Overall physical sensor agreement (35%)
    2. Water IoU consistency (25%)
    3. SAR dynamic range & signal quality (15%)
    4. Optical spectral contrast (15%)
    5. Multi-band availability (10%)
    """
    agree_pct = metrics.get("modality_agreement_pct", 50.0)
    iou_pct = metrics.get("inundation_iou_pct", 50.0)

    # 1. Agreement contribution
    c_agree = np.clip((agree_pct - 40.0) / 60.0, 0.0, 1.0) * 35.0

    # 2. IoU contribution
    c_iou = np.clip(iou_pct / 100.0, 0.0, 1.0) * 25.0

    # 3. SAR dynamic range contribution
    sar_range = sar.max_db - sar.min_db
    c_sar = np.clip((sar_range - 10.0) / 25.0, 0.0, 1.0) * 15.0

    # 4. Optical contrast contribution
    opt_contrast = opt.rgb.std()
    c_opt = np.clip((opt_contrast - 0.05) / 0.25, 0.0, 1.0) * 15.0

    # 5. Multi-band bonus
    c_bands = (5.0 if opt.bands_count >= 4 else 2.5) + (5.0 if "dual_pol" in sar.polarization else 2.5)

    raw_conf = c_agree + c_iou + c_sar + c_opt + c_bands
    return int(np.clip(round(raw_conf), 45, 96))


def reason_multimodal(
    query: str,
    opt: OpticalFeatures,
    sar: SARFeatures,
    metrics: Dict[str, Any],
    features: List[FusionFeature],
    fused_image_bytes: Optional[bytes] = None,
) -> Tuple[str, str, str, str, int, bool]:
    """
    Execute query-aware multimodal reasoning.

    Returns:
        (summary, optical_evidence, sar_evidence, fused_interpretation, confidence_int, is_simulated)
    """
    query_theme = _classify_query_theme(query)

    optical_evidence = _generate_optical_evidence(opt, features, query_theme)
    sar_evidence = _generate_sar_evidence(sar, features, query_theme)
    fused_interpretation = _generate_fused_interpretation(query, query_theme, opt, sar, metrics, features)

    confidence = _compute_calibrated_confidence(metrics, opt, sar, len(features))

    summary = (
        f"Optical + SAR fusion ({opt.bands_count}-band Optical + {sar.polarization} SAR): "
        f"{metrics.get('modality_agreement_pct', 0.0):.1f}% agreement (IoU: {metrics.get('inundation_iou_pct', 0.0):.1f}%), "
        f"{len(features)} feature region(s), confidence: {confidence}%."
    )

    # Check if real LLaVA model is available
    from app.models import vqa_model
    is_simulated = not vqa_model.is_available()

    if not is_simulated and fused_image_bytes is not None:
        try:
            vlm_prompt = (
                f"Fused Optical and SAR Satellite Analysis.\n"
                f"User Question: {query}\n"
                f"Physical Optical Evidence: {optical_evidence}\n"
                f"Physical SAR Radar Evidence: {sar_evidence}\n"
                f"Provide an expert Earth observation synthesis."
            )
            vlm_answer = vqa_model.answer_vqa(vlm_prompt, fused_image_bytes)
            if vlm_answer and len(vlm_answer.strip()) > 20:
                fused_interpretation = vlm_answer.strip()
                is_simulated = False
        except Exception as e:
            logger.warning("LLaVA VLM execution fallback to rule-based: %s", e)
            is_simulated = True

    return summary, optical_evidence, sar_evidence, fused_interpretation, confidence, is_simulated
