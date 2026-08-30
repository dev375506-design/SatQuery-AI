
from __future__ import annotations

import logging
from typing import List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.agent.router import ImageInput, run_agent
from app.models import vqa_model
from app.schemas import (
    AnalyzeResponse,
    ChangeAnalysisResponse,
    FusionAnalysisResponse,
    HealthResponse,
)


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="SatQuery AI Backend",
    version="0.1.0",
    description="Backend API for SatQuery AI remote-sensing image analysis.",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
# Wide open for local prototype development.
# Restrict this before production deployment.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Helper: normalize optional uploads
# ---------------------------------------------------------

def _normalize_upload(
    value: Union[UploadFile, str, None]
) -> Optional[UploadFile]:
    """
    Swagger/browser may send an empty optional file field as an empty string.

    Convert:
        None       -> None
        ""         -> None
        UploadFile -> UploadFile

    This prevents a 422 validation error when image2 is left empty.
    """

    if value is None:
        return None

    if isinstance(value, str):
        return None

    return value


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    """
    Check whether the backend is running and whether
    the VQA model is available.
    """

    return HealthResponse(
        status="ok",
        vqa_model_loaded=vqa_model.is_available(),
        device=config.DEVICE,
    )


# ---------------------------------------------------------
# Main analysis endpoint
# ---------------------------------------------------------

@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
)
async def analyze(
    query: str = Form(...),
    image1: Union[UploadFile, str, None] = File(default=None),
    image2: Union[UploadFile, str, None] = File(default=None),
):
    """
    Analyze a remote-sensing query using one or two images.

    Single-image example:
        query + image1

    Two-image example:
        query + image1 + image2

    If Swagger sends an empty string for image2,
    it is treated as None.
    """

    # Normalize optional upload fields.
    image1 = _normalize_upload(image1)
    image2 = _normalize_upload(image2)

    images: List[ImageInput] = []

    # Process image1 and image2.
    for uploaded_file in (image1, image2):

        if uploaded_file is None:
            continue

        data = await uploaded_file.read()

        # Ignore empty uploaded files.
        if not data:
            logger.warning(
                "Received empty image file: %s",
                uploaded_file.filename,
            )
            continue

        images.append(
            ImageInput(
                filename=uploaded_file.filename or "uploaded_image",
                content_type=uploaded_file.content_type
                or "application/octet-stream",
                data=data,
            )
        )

    logger.info(
        "Analysis request received: query=%r, images=%d",
        query,
        len(images),
    )

    # Send request to the agent/router.
    result = run_agent(
        query=query,
        images=images,
    )

    return result


# ---------------------------------------------------------
# Bi-Temporal Change Analysis endpoint
# ---------------------------------------------------------

@app.post(
    "/api/analyze/change",
    response_model=ChangeAnalysisResponse,
)
async def analyze_change(
    query: str = Form(default="What changed between these images?"),
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
):
    """
    Bi-temporal change analysis.

    Compares two satellite images (before/after) and returns:
    - Change map visualisation
    - Change overlay
    - Detected change regions
    - AI-generated explanation
    - Execution trace
    """

    # Read image data
    before_data = await before_image.read()
    after_data = await after_image.read()

    if not before_data:
        raise HTTPException(
            status_code=400,
            detail="Before image is empty or missing.",
        )
    if not after_data:
        raise HTTPException(
            status_code=400,
            detail="After image is empty or missing.",
        )

    logger.info(
        "Change analysis request: query=%r, before=%s (%d bytes), after=%s (%d bytes)",
        query,
        before_image.filename,
        len(before_data),
        after_image.filename,
        len(after_data),
    )

    try:
        from app.change_analysis.pipeline import run_change_analysis

        result = run_change_analysis(
            before_bytes=before_data,
            after_bytes=after_data,
            query=query,
        )

        # Convert internal result to API response
        from app.schemas import BoundingBoxResponse, ChangeRegionResponse

        return ChangeAnalysisResponse(
            task=result.task,
            summary=result.summary,
            query=result.query,
            changes=[
                ChangeRegionResponse(
                    region_id=r.region_id,
                    bbox=BoundingBoxResponse(
                        x=r.bbox.x,
                        y=r.bbox.y,
                        width=r.bbox.width,
                        height=r.bbox.height,
                    ),
                    area_pixels=r.area_pixels,
                    area_fraction=r.area_fraction,
                    location_description=r.location_description,
                    mean_magnitude=r.mean_magnitude,
                    confidence=r.confidence,
                )
                for r in result.changes
            ],
            changed_area_percentage=result.changed_area_percentage,
            overall_confidence=result.overall_confidence,
            before_image=result.before_image_b64,
            after_image=result.after_image_b64,
            change_map=result.change_map_b64,
            change_overlay=result.change_overlay_b64,
            execution_trace=result.execution_trace,
            simulated=result.simulated,
            geo_crs=result.geo_crs,
            geo_bounds=result.geo_bounds,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("Change analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"Change analysis error: {e}",
        )


# ---------------------------------------------------------
# Optical + SAR Fusion endpoint
# ---------------------------------------------------------

@app.post(
    "/api/analyze/fusion",
    response_model=FusionAnalysisResponse,
)
async def analyze_fusion(
    query: str = Form(default="Analyze this scene using fused optical and SAR satellite imagery."),
    optical_image: UploadFile = File(...),
    sar_image: UploadFile = File(...),
    fusion_method: Optional[str] = Form(default="composite"),
):
    """
    Optical + SAR Multimodal Fusion.

    Fuses co-registered optical and synthetic aperture radar (SAR) imagery,
    returning:
    - Preprocessed optical preview
    - Log-scaled radar backscatter preview
    - False-color / IHS fused composite visualization
    - Quantitative cross-modal agreement heatmap
    - Separated optical evidence, SAR evidence, and fused interpretation
    - Localized cross-modal features
    - Execution trace
    """
    optical_data = await optical_image.read()
    sar_data = await sar_image.read()

    if not optical_data:
        raise HTTPException(
            status_code=400,
            detail="Optical image is empty or missing.",
        )
    if not sar_data:
        raise HTTPException(
            status_code=400,
            detail="SAR image is empty or missing.",
        )

    logger.info(
        "Fusion request: query=%r, opt=%s (%d bytes), sar=%s (%d bytes), method=%s",
        query,
        optical_image.filename,
        len(optical_data),
        sar_image.filename,
        len(sar_data),
        fusion_method,
    )

    try:
        from app.fusion.pipeline import run_fusion_analysis

        result = run_fusion_analysis(
            optical_bytes=optical_data,
            sar_bytes=sar_data,
            query=query,
            fusion_method=fusion_method or "composite",
        )

        from app.schemas import BoundingBoxResponse, FusionFeatureRegion

        return FusionAnalysisResponse(
            task=result.task,
            query=result.query,
            summary=result.summary,
            optical_evidence=result.optical_evidence,
            sar_evidence=result.sar_evidence,
            fused_interpretation=result.fused_interpretation,
            confidence=result.confidence,
            modality_agreement_percentage=result.modality_agreement_percentage,
            optical_image=result.optical_image_b64,
            sar_image=result.sar_image_b64,
            fusion_visualization=result.fusion_visualization_b64,
            evidence_map=result.evidence_map_b64,
            features=[
                FusionFeatureRegion(
                    id=f.feature_id,
                    bbox=BoundingBoxResponse(
                        x=f.bbox.x,
                        y=f.bbox.y,
                        width=f.bbox.width,
                        height=f.bbox.height,
                    ),
                    category=f.category,
                    optical_characteristics=f.optical_characteristics,
                    sar_characteristics=f.sar_characteristics,
                    agreement_score=f.agreement_score,
                    location_description=f.location_description,
                    confidence=f.confidence,
                    optical_score=getattr(f, "optical_score", None),
                    sar_score=getattr(f, "sar_score", None),
                    region_area_pixels=getattr(f, "region_area_pixels", None),
                    region_area_pct=getattr(f, "region_area_pct", None),
                    classification=getattr(f, "classification", f.category),
                )
                for f in result.features
            ],

            alignment_method=result.alignment_info.method if result.alignment_info else "dimension_matched",
            alignment_details=result.alignment_info.details if result.alignment_info else "",
            diagnostics=result.diagnostics,
            execution_trace=result.execution_trace,
            simulated=result.simulated,
            geo_crs=result.geo_crs,
            geo_bounds=result.geo_bounds,
        )


    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("Fusion analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"Fusion analysis error: {e}",
        )
