"""
Schemas for the Optical + SAR multimodal fusion pipeline.

These are internal dataclasses used across the fusion submodules.
The API-facing Pydantic models live in app.schemas (FusionAnalysisResponse).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class BoundingBox:
    """Pixel-coordinate bounding box for a localized region."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class FusionFeature:
    """A localized cross-modal feature characterized by both sensors."""
    feature_id: int
    bbox: BoundingBox
    category: str  # e.g., "Surface Water / Inundation Candidate", "Built-up Infrastructure / Settlement"
    optical_characteristics: str
    sar_characteristics: str
    agreement_score: float  # 0..1
    location_description: str  # e.g. "central sector"
    confidence: float  # 0..1
    optical_score: float = 0.0
    sar_score: float = 0.0
    region_area_pixels: int = 0
    region_area_pct: float = 0.0
    classification: str = ""


@dataclass
class OpticalFeatures:
    """Extracted physical and statistical features from optical imagery."""
    rgb: np.ndarray  # (H, W, 3) float32 [0, 1] for visual display
    b02_blue: np.ndarray  # (H, W) float32 reflectance [0, 1]
    b03_green: np.ndarray  # (H, W) float32 reflectance [0, 1]
    b04_red: np.ndarray  # (H, W) float32 reflectance [0, 1]
    b08_nir: Optional[np.ndarray]  # (H, W) float32 reflectance [0, 1] (if 4+ bands)
    ndwi: np.ndarray  # (H, W) float32 [-1, 1] Normalized Difference Water Index
    ndvi: Optional[np.ndarray]  # (H, W) float32 [-1, 1] Normalized Difference Vegetation Index
    luminance: np.ndarray  # (H, W) float32 [0, 1]
    edges: np.ndarray  # (H, W) float32 [0, 1]
    water_mask: np.ndarray  # (H, W) bool
    vegetation_mask: np.ndarray  # (H, W) bool
    mean_brightness: float
    ndwi_threshold: float = 0.05
    bands_count: int = 3
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SARFeatures:
    """Extracted radar backscatter and structural features from SAR imagery."""
    vv_db: np.ndarray  # (H, W) float32 in decibels (e.g. -35 to +5 dB)
    vh_db: Optional[np.ndarray]  # (H, W) float32 in decibels (if dual-pol)
    vv_vh_ratio_db: Optional[np.ndarray]  # (H, W) float32 cross-pol ratio (VV_dB - VH_dB)
    db_map_norm: np.ndarray  # (H, W) float32 log-scaled [0, 1]
    double_bounce_mask: np.ndarray  # (H, W) bool (high backscatter structures)
    specular_mask: np.ndarray  # (H, W) bool (low backscatter water candidates)
    volume_scatter_mask: Optional[np.ndarray]  # (H, W) bool (high cross-pol)
    water_threshold_db: float = -14.0
    double_bounce_threshold_db: float = -8.0
    mean_db: float = 0.0  # physical mean in dB
    min_db: float = 0.0
    max_db: float = 0.0
    std_db: float = 0.0
    polarization: str = "single_pol"  # "single_pol (VV)", "dual_pol (VV+VH)"
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentInfo:
    """Result and metadata of spatial alignment between optical and SAR inputs."""
    method: str  # "geo_referenced" or "dimension_matched"
    details: str
    target_shape: Tuple[int, int]
    crs: Optional[str] = None
    bounds: Optional[str] = None
    geo_used: bool = False
    resolution_m: Optional[Tuple[float, float]] = None
    optical_crs: Optional[str] = None
    sar_crs: Optional[str] = None
    optical_bounds: Optional[str] = None
    sar_bounds: Optional[str] = None
    common_crs: Optional[str] = None
    common_grid_resolution: Optional[str] = None


@dataclass
class FusionAnalysisResult:
    """Complete result returned by the Optical + SAR fusion pipeline."""
    task: str = "optical_sar_fusion"
    query: str = ""
    summary: str = ""

    optical_evidence: str = ""
    sar_evidence: str = ""
    fused_interpretation: str = ""

    confidence: int = 0  # 0..100
    modality_agreement_percentage: float = 0.0

    optical_image_b64: str = ""
    sar_image_b64: str = ""
    fusion_visualization_b64: str = ""
    evidence_map_b64: Optional[str] = None

    features: List[FusionFeature] = field(default_factory=list)
    alignment_info: Optional[AlignmentInfo] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    execution_trace: List[str] = field(default_factory=list)
    simulated: bool = False

    geo_crs: Optional[str] = None
    geo_bounds: Optional[str] = None
