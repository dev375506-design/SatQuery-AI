"""
Response/request schemas.

IMPORTANT: `AnalyzeResponse` is intentionally shaped to match what the existing
frontend (script.js -> DEMOS[key].render / showResults) already expects:
  - task            -> demo.task           (shown in resultTaskTag)
  - model           -> demo.model          (shown in metaModel)
  - answer          -> the main answer text
  - confidence      -> demo.confidence     (0-100 int, shown in conf bar)
  - trace           -> demo.trace          (list[str], shown as execution steps)
  - visual_evidence -> optional base64 PNG/text describing highlighted regions
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    title: str
    detail: Optional[str] = None


class AnalyzeResponse(BaseModel):
    task: str = Field(..., description="Human-readable task name, e.g. 'Single-Image VQA'")
    model: str = Field(..., description="Model/tool name that produced the answer")
    query: str
    answer: str
    confidence: int = Field(..., ge=0, le=100)
    trace: List[str] = Field(default_factory=list)
    visual_evidence: Optional[str] = Field(
        default=None, description="Optional base64-encoded PNG with overlays, or null"
    )
    simulated: bool = Field(
        default=True,
        description="True if this response came from the fallback/simulated path "
                     "rather than the real fine-tuned model (model not loaded/available).",
    )


# -----------------------------------------------------------------
# Bi-Temporal Change Analysis response
# -----------------------------------------------------------------

class BoundingBoxResponse(BaseModel):
    x: int
    y: int
    width: int
    height: int


class ChangeRegionResponse(BaseModel):
    region_id: int
    bbox: BoundingBoxResponse
    area_pixels: int
    area_fraction: float
    location_description: str
    mean_magnitude: float
    confidence: float


class ChangeAnalysisResponse(BaseModel):
    """
    Response for POST /api/analyze/change.

    Contains all visual outputs (base64-encoded PNGs), detected change
    regions, AI interpretation, and the execution trace.
    """
    task: str = Field(default="bi_temporal_change_analysis")
    summary: str = Field(default="", description="AI-generated explanation of changes")
    query: str = ""

    changes: List[ChangeRegionResponse] = Field(default_factory=list)
    changed_area_percentage: float = 0.0
    overall_confidence: float = 0.0

    before_image: str = Field(default="", description="Base64 PNG of the before image")
    after_image: str = Field(default="", description="Base64 PNG of the after image")
    change_map: str = Field(default="", description="Base64 PNG heatmap of change magnitude")
    change_overlay: str = Field(default="", description="Base64 PNG of after image with change highlights")

    execution_trace: List[str] = Field(default_factory=list)
    simulated: bool = True

    geo_crs: Optional[str] = None
    geo_bounds: Optional[str] = None


# -----------------------------------------------------------------
# Optical + SAR Fusion response
# -----------------------------------------------------------------

class FusionFeatureRegion(BaseModel):
    id: int
    bbox: BoundingBoxResponse
    category: str
    optical_characteristics: str
    sar_characteristics: str
    agreement_score: float
    location_description: str
    confidence: float
    optical_score: Optional[float] = None
    sar_score: Optional[float] = None
    region_area_pixels: Optional[int] = None
    region_area_pct: Optional[float] = None
    classification: Optional[str] = None


class FusionAnalysisResponse(BaseModel):
    """
    Response for POST /api/analyze/fusion.

    Contains all visual outputs (base64-encoded PNGs of optical, SAR, fused composite,
    evidence map), separate optical and SAR evidence, cross-modal interpretation,
    detected features, alignment metadata, and execution trace.
    """
    task: str = Field(default="optical_sar_fusion")
    query: str = ""
    summary: str = Field(default="", description="Executive summary of the fused analysis")

    optical_evidence: str = Field(default="", description="Direct optical observations")
    sar_evidence: str = Field(default="", description="Direct SAR radar backscatter observations")
    fused_interpretation: str = Field(default="", description="Cross-modal synthesis and interpretation")

    confidence: int = Field(default=0, ge=0, le=100, description="Confidence score 0-100 based on cross-modal agreement")
    modality_agreement_percentage: float = 0.0

    optical_image: str = Field(default="", description="Base64 PNG of optical image")
    sar_image: str = Field(default="", description="Base64 PNG of SAR image (log-scaled backscatter)")
    fusion_visualization: str = Field(default="", description="Base64 PNG of fused composite")
    evidence_map: Optional[str] = Field(default=None, description="Base64 PNG of cross-modal agreement heatmap")

    features: List[FusionFeatureRegion] = Field(default_factory=list)
    alignment_method: str = Field(default="dimension_matched", description="'geo_referenced' or 'dimension_matched'")
    alignment_details: str = Field(default="", description="Details on spatial alignment performed")

    diagnostics: Optional[Dict[str, Any]] = Field(default=None, description="Detailed physical and scientific band diagnostics")
    execution_trace: List[str] = Field(default_factory=list)
    simulated: bool = True

    geo_crs: Optional[str] = None
    geo_bounds: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    vqa_model_loaded: bool
    device: str
