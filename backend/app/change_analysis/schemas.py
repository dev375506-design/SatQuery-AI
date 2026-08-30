"""
Schemas for the bi-temporal change analysis pipeline.

These are internal dataclasses used by the pipeline.  The API-facing Pydantic
models live in app.schemas (ChangeAnalysisResponse).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BoundingBox:
    """Pixel-coordinate bounding box for a change region."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class ChangeRegion:
    """A single detected change region."""
    region_id: int
    bbox: BoundingBox
    area_pixels: int
    area_fraction: float          # fraction of total image area
    location_description: str     # e.g. "upper-left quadrant"
    mean_magnitude: float         # average difference magnitude 0..1
    confidence: float             # 0..1


@dataclass
class ChangeAnalysisResult:
    """Full result returned by the change analysis pipeline."""
    task: str = "bi_temporal_change_analysis"
    summary: str = ""
    query: str = ""

    changes: List[ChangeRegion] = field(default_factory=list)
    changed_area_percentage: float = 0.0
    overall_confidence: float = 0.0

    before_image_b64: str = ""    # base64 PNG
    after_image_b64: str = ""
    change_map_b64: str = ""
    change_overlay_b64: str = ""

    execution_trace: List[str] = field(default_factory=list)
    simulated: bool = False

    # Optional geo metadata (populated only when GeoTIFF headers exist)
    geo_crs: Optional[str] = None
    geo_bounds: Optional[str] = None
