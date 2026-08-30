"""
Multimodal Fusion Engine for Optical and SAR satellite imagery.

Combines calibrated physical reflectance and radar backscatter properties:
1. Calculates pixel-level cross-modal spatial agreement and IoU consensus.
2. Generates false-color Optical-SAR and IHS composites.
3. Extracts discrete, localized cross-modal feature regions with evidence scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np

from app.fusion.schemas import (
    BoundingBox,
    FusionFeature,
    OpticalFeatures,
    SARFeatures,
)

logger = logging.getLogger("satquery.fusion.fusion_engine")


@dataclass
class FusedData:
    """Internal multimodal fused representation and agreement maps."""
    composite_rgb: np.ndarray  # (H, W, 3) float32 [0, 1]
    ihs_fused_rgb: np.ndarray  # (H, W, 3) float32 [0, 1]
    agreement_heatmap: np.ndarray  # (H, W, 3) float32 [0, 1]
    water_agreement_mask: np.ndarray  # (H, W) bool
    structure_agreement_mask: np.ndarray  # (H, W) bool
    overall_agreement_pct: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    features: List[FusionFeature] = field(default_factory=list)


def _location_description(cx: float, cy: float) -> str:
    """Describe approximate region location in the image grid."""
    v = "northern" if cy < 0.33 else ("southern" if cy > 0.66 else "central")
    h = "western" if cx < 0.33 else ("eastern" if cx > 0.66 else "central")
    if v == "central" and h == "central":
        return "center of the scene"
    if v == "central":
        return f"{h} sector"
    if h == "central":
        return f"{v} sector"
    return f"{v}-{h} quadrant"


def _connected_components(mask: np.ndarray) -> np.ndarray:
    """Label connected components in a 2D boolean mask."""
    try:
        from scipy.ndimage import label as scipy_label
        labels, _ = scipy_label(mask)
        return labels
    except ImportError:
        labels = np.zeros_like(mask, dtype=np.int32)
        cur = 0
        h, w = mask.shape
        for y in range(h):
            for x in range(w):
                if mask[y, x] and labels[y, x] == 0:
                    cur += 1
                    stack = [(y, x)]
                    while stack:
                        cy, cx = stack.pop()
                        if cy < 0 or cy >= h or cx < 0 or cx >= w or not mask[cy, cx] or labels[cy, cx] != 0:
                            continue
                        labels[cy, cx] = cur
                        stack.extend([(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)])
        return labels


class FusionEngine:
    """
    Multimodal fusion engine combining optical imagery with radar backscatter.
    """

    def __init__(self, method: str = "composite"):
        self.method = method

    def fuse(
        self,
        opt: OpticalFeatures,
        sar: SARFeatures,
        min_feature_area_pct: float = 0.0005,
        max_features: int = 15,
    ) -> FusedData:
        """
        Execute cross-modal fusion on calibrated optical and SAR representations.
        """
        h, w = opt.luminance.shape
        total_pixels = h * w

        # -------------------------------------------------------------
        # 1. Cross-Modal Spatial Agreement Analysis
        # -------------------------------------------------------------
        water_opt = opt.water_mask
        water_sar = sar.specular_mask

        # Consensus Surface Inundation Candidate: BOTH sensors detect water signature
        consensus_water = water_opt & water_sar

        # Disagreements
        opt_only_water = water_opt & (~water_sar)
        sar_only_water = (~water_opt) & water_sar

        # Consensus Dry Non-Inundated Land: BOTH agree surface is not water
        consensus_dry_land = (~water_opt) & (~water_sar)

        # Structure Consensus: Optical edges AND SAR double-bounce backscatter
        structure_opt = opt.edges > 0.20
        structure_sar = sar.double_bounce_mask
        consensus_structure = structure_opt & structure_sar

        # Exact Mathematical Agreement Calculations
        opt_water_pct = round(float(water_opt.sum()) / total_pixels * 100.0, 2)
        sar_water_pct = round(float(water_sar.sum()) / total_pixels * 100.0, 2)
        consensus_water_pct = round(float(consensus_water.sum()) / total_pixels * 100.0, 2)
        opt_only_water_pct = round(float(opt_only_water.sum()) / total_pixels * 100.0, 2)
        sar_only_water_pct = round(float(sar_only_water.sum()) / total_pixels * 100.0, 2)
        consensus_dry_pct = round(float(consensus_dry_land.sum()) / total_pixels * 100.0, 2)

        # Modality Agreement % = (Consensus Water + Consensus Dry Land) / Total Overlap
        overall_agreement_pct = round(consensus_water_pct + consensus_dry_pct, 2)
        disagreement_pct = round(opt_only_water_pct + sar_only_water_pct, 2)

        # Water Inundation IoU (Jaccard Index)
        union_water_count = (water_opt | water_sar).sum()
        inundation_iou_pct = round(
            float(consensus_water.sum()) / max(1, union_water_count) * 100.0, 2
        ) if union_water_count > 0 else 100.0
        total_candidate_inundation_pct = round(float(union_water_count) / total_pixels * 100.0, 2)

        metrics: Dict[str, Any] = {
            "valid_overlap_pct": 100.0,
            "optical_water_candidate_pct": opt_water_pct,
            "sar_water_candidate_pct": sar_water_pct,
            "optical_sar_consensus_water_pct": consensus_water_pct,
            "optical_only_water_pct": opt_only_water_pct,
            "sar_only_water_pct": sar_only_water_pct,
            "consensus_dry_land_pct": consensus_dry_pct,
            "modality_agreement_pct": overall_agreement_pct,
            "inundation_iou_pct": inundation_iou_pct,
            "disagreement_pct": disagreement_pct,
            "total_candidate_inundation_area_pct": total_candidate_inundation_pct,
            "permanent_water_handling_status": "single_date_candidate (no historical temporal baseline provided)",
        }

        # -------------------------------------------------------------
        # 2. False-Color Optical-SAR Composite (R, G, B)
        # -------------------------------------------------------------
        comp_r = np.clip(opt.b04_red * 0.65 + (sar.double_bounce_mask * 0.35), 0.0, 1.0)
        comp_g = np.clip(opt.b03_green * 0.85 + (opt.vegetation_mask * 0.15), 0.0, 1.0)
        comp_b = np.clip(sar.db_map_norm * 0.70 + (opt.b02_blue * 0.30), 0.0, 1.0)
        composite_rgb = np.stack([comp_r, comp_g, comp_b], axis=-1).astype(np.float32)

        # -------------------------------------------------------------
        # 3. IHS-based Spatial Texture Modulation
        # -------------------------------------------------------------
        sar_hf = sar.db_map_norm - float(sar.db_map_norm.mean())
        modulated_lum = np.clip(opt.luminance + 0.35 * sar_hf, 0.0, 1.0)
        lum_ratio = np.where(opt.luminance > 0.01, modulated_lum / (opt.luminance + 1e-5), 1.0)
        lum_ratio = np.clip(lum_ratio, 0.3, 2.0)
        ihs_rgb = np.clip(opt.rgb * lum_ratio[..., np.newaxis], 0.0, 1.0).astype(np.float32)

        # -------------------------------------------------------------
        # 4. Quantitative Agreement Heatmap
        # -------------------------------------------------------------
        agreement_hm = np.zeros((h, w, 3), dtype=np.float32)
        base_gray = opt.luminance * 0.25
        agreement_hm[..., 0] = base_gray
        agreement_hm[..., 1] = base_gray
        agreement_hm[..., 2] = base_gray

        # Water consensus (Cyan: Optical NDWI + SAR Specular confirm inundation candidate)
        agreement_hm[consensus_water, 0] = 0.05
        agreement_hm[consensus_water, 1] = 0.78
        agreement_hm[consensus_water, 2] = 0.98

        # Structure consensus (Magenta / Red)
        agreement_hm[consensus_structure, 0] = 0.98
        agreement_hm[consensus_structure, 1] = 0.20
        agreement_hm[consensus_structure, 2] = 0.50

        # Flooded vegetation discrepancy (Yellow: Optical Vegetation + SAR sub-canopy water)
        flooded_veg_mask = opt.vegetation_mask & water_sar
        agreement_hm[flooded_veg_mask, 0] = 0.95
        agreement_hm[flooded_veg_mask, 1] = 0.85
        agreement_hm[flooded_veg_mask, 2] = 0.10

        # -------------------------------------------------------------
        # 5. Extract Distinct Cross-Modal Feature Regions
        # -------------------------------------------------------------
        features: List[FusionFeature] = []
        min_pixels = max(15, int(total_pixels * min_feature_area_pct))
        fid = 1

        # Region Category 1: Surface Water / Inundation Candidate
        water_labels = _connected_components(consensus_water if consensus_water.sum() > 0 else (water_opt | water_sar))
        for rid in np.unique(water_labels):
            if rid == 0:
                continue
            rmask = water_labels == rid
            area = int(rmask.sum())
            if area < min_pixels:
                continue
            ys, xs = np.where(rmask)
            ymin, ymax = int(ys.min()), int(ys.max())
            xmin, xmax = int(xs.min()), int(xs.max())
            cx, cy = float(xs.mean()) / w, float(ys.mean()) / h

            opt_ndwi_val = float(opt.ndwi[rmask].mean())
            sar_vv_val = float(sar.vv_db[rmask].mean())

            # Specific evidence scores [0, 1]
            opt_score = min(1.0, max(0.0, (opt_ndwi_val - opt.ndwi_threshold + 0.2) / 0.6))
            sar_score = min(1.0, max(0.0, (sar.water_threshold_db - sar_vv_val + 5.0) / 15.0))
            cross_agree = float(consensus_water[rmask].sum()) / area if area > 0 else 0.5

            region_conf = round(0.35 * opt_score + 0.35 * sar_score + 0.30 * cross_agree, 2)
            region_area_pct = round(float(area) / total_pixels * 100.0, 3)

            features.append(FusionFeature(
                feature_id=fid,
                bbox=BoundingBox(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1),
                category="Surface Water / Inundation Candidate",
                optical_characteristics=f"Mean NDWI {opt_ndwi_val:+.3f} with NIR absorption",
                sar_characteristics=f"Low radar backscatter {sar_vv_val:.1f} dB (specular reflection)",
                agreement_score=round(cross_agree, 2),
                location_description=_location_description(cx, cy),
                confidence=region_conf,
                optical_score=round(opt_score, 2),
                sar_score=round(sar_score, 2),
                region_area_pixels=area,
                region_area_pct=region_area_pct,
                classification="Surface Water / Inundation Candidate",
            ))
            fid += 1

        # Region Category 2: Built-up Infrastructure / Settlement
        struct_labels = _connected_components(consensus_structure)
        for rid in np.unique(struct_labels):
            if rid == 0:
                continue
            rmask = struct_labels == rid
            area = int(rmask.sum())
            if area < min_pixels:
                continue
            ys, xs = np.where(rmask)
            ymin, ymax = int(ys.min()), int(ys.max())
            xmin, xmax = int(xs.min()), int(xs.max())
            cx, cy = float(xs.mean()) / w, float(ys.mean()) / h

            sar_vv_val = float(sar.vv_db[rmask].mean())
            opt_edge_val = float(opt.edges[rmask].mean())

            opt_score = min(1.0, max(0.0, opt_edge_val / 0.5))
            sar_score = min(1.0, max(0.0, (sar_vv_val - sar.double_bounce_threshold_db + 5.0) / 10.0))
            cross_agree = float(consensus_structure[rmask].sum()) / area if area > 0 else 0.5
            region_conf = round(0.35 * opt_score + 0.35 * sar_score + 0.30 * cross_agree, 2)
            region_area_pct = round(float(area) / total_pixels * 100.0, 3)

            features.append(FusionFeature(
                feature_id=fid,
                bbox=BoundingBox(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1),
                category="Built-up Infrastructure / Settlement",
                optical_characteristics=f"High spatial edge density ({opt_edge_val:.2f})",
                sar_characteristics=f"Double-bounce backscatter ({sar_vv_val:.1f} dB)",
                agreement_score=round(cross_agree, 2),
                location_description=_location_description(cx, cy),
                confidence=region_conf,
                optical_score=round(opt_score, 2),
                sar_score=round(sar_score, 2),
                region_area_pixels=area,
                region_area_pct=region_area_pct,
                classification="Built-up Infrastructure / Settlement",
            ))
            fid += 1

        # Region Category 3: Flooded Agricultural Canopy / Wetland (Rigorous Overlap Requirement)
        flooded_veg_labels = _connected_components(flooded_veg_mask)
        for rid in np.unique(flooded_veg_labels):
            if rid == 0:
                continue
            rmask = flooded_veg_labels == rid
            area = int(rmask.sum())
            if area < min_pixels:
                continue
            ys, xs = np.where(rmask)
            ymin, ymax = int(ys.min()), int(ys.max())
            xmin, xmax = int(xs.min()), int(xs.max())
            cx, cy = float(xs.mean()) / w, float(ys.mean()) / h

            ndvi_val = float(opt.ndvi[rmask].mean()) if opt.ndvi is not None else 0.35
            sar_vv_val = float(sar.vv_db[rmask].mean())

            opt_score = min(1.0, max(0.0, ndvi_val / 0.6))
            sar_score = min(1.0, max(0.0, (sar.water_threshold_db - sar_vv_val + 3.0) / 10.0))
            cross_agree = float(flooded_veg_mask[rmask].sum()) / area if area > 0 else 0.5
            region_conf = round(0.35 * opt_score + 0.35 * sar_score + 0.30 * cross_agree, 2)
            region_area_pct = round(float(area) / total_pixels * 100.0, 3)

            features.append(FusionFeature(
                feature_id=fid,
                bbox=BoundingBox(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1),
                category="Flooded Agricultural Canopy / Wetland",
                optical_characteristics=f"Active vegetation response (NDVI {ndvi_val:.2f})",
                sar_characteristics=f"Sub-canopy radar attenuation ({sar_vv_val:.1f} dB)",
                agreement_score=round(cross_agree, 2),
                location_description=_location_description(cx, cy),
                confidence=region_conf,
                optical_score=round(opt_score, 2),
                sar_score=round(sar_score, 2),
                region_area_pixels=area,
                region_area_pct=region_area_pct,
                classification="Flooded Agricultural Canopy / Wetland",
            ))
            fid += 1

        # Sort features by bounding box area descending and limit to max_features
        features.sort(key=lambda f: f.bbox.width * f.bbox.height, reverse=True)
        if len(features) > max_features:
            features = features[:max_features]

        metrics["number_of_regions"] = len(features)

        logger.info(
            "Fusion complete: physical agreement=%.2f%%, IoU=%.2f%%, candidate inundation=%.2f%%, features=%d",
            overall_agreement_pct, inundation_iou_pct, total_candidate_inundation_pct, len(features)
        )

        return FusedData(
            composite_rgb=composite_rgb,
            ihs_fused_rgb=ihs_rgb,
            agreement_heatmap=agreement_hm,
            water_agreement_mask=consensus_water,
            structure_agreement_mask=consensus_structure,
            overall_agreement_pct=overall_agreement_pct,
            metrics=metrics,
            features=features,
        )
