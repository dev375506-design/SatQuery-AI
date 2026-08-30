"""
Bi-temporal change analysis pipeline.

Orchestrates the full flow:
    validate → preprocess → detect → extract regions →
    visualise → interpret → assemble result

This is the single entry point called by the API endpoint.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Optional

from app.change_analysis.detector import ChangeDetector
from app.change_analysis.interpreter import interpret_changes
from app.change_analysis.preprocessing import (
    load_and_normalise,
    make_compatible,
)
from app.change_analysis.region_extractor import extract_regions
from app.change_analysis.schemas import ChangeAnalysisResult
from app.change_analysis.visualization import (
    generate_change_map,
    generate_change_overlay,
    image_array_to_base64,
)

logger = logging.getLogger("satquery.change.pipeline")


def run_change_analysis(
    before_bytes: bytes,
    after_bytes: bytes,
    query: str = "",
) -> ChangeAnalysisResult:
    """
    Execute the full bi-temporal change analysis pipeline.

    Parameters
    ----------
    before_bytes : raw bytes of the "before" / earlier image
    after_bytes  : raw bytes of the "after" / later image
    query        : optional natural-language question

    Returns
    -------
    ChangeAnalysisResult with all outputs populated.
    """
    result = ChangeAnalysisResult(query=query or "What changed between these images?")
    trace = result.execution_trace
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    trace.append("Input validation started")

    if not before_bytes:
        raise ValueError("Before image is empty or missing.")
    if not after_bytes:
        raise ValueError("After image is empty or missing.")

    trace.append("✓ Both images present and non-empty")

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    trace.append("Image preprocessing started")

    before_arr, geo_before = load_and_normalise(before_bytes)
    after_arr, geo_after = load_and_normalise(after_bytes)

    trace.append(
        f"✓ Before image loaded: {before_arr.shape[1]}×{before_arr.shape[0]} px"
    )
    trace.append(
        f"✓ After image loaded: {after_arr.shape[1]}×{after_arr.shape[0]} px"
    )

    # Preserve geo metadata if available
    geo = geo_before or geo_after
    if geo:
        result.geo_crs = geo.crs
        result.geo_bounds = geo.bounds
        trace.append(f"✓ GeoTIFF metadata found: CRS={geo.crs}")
    else:
        trace.append("  No GeoTIFF metadata (using pixel coordinates)")

    # ------------------------------------------------------------------
    # 3. Image alignment / dimension matching
    # ------------------------------------------------------------------
    trace.append("Image alignment / dimension matching")

    before_arr, after_arr = make_compatible(before_arr, after_arr)

    trace.append(
        f"✓ Images aligned to {before_arr.shape[1]}×{before_arr.shape[0]} px"
    )

    # Save base64 versions for the response
    result.before_image_b64 = image_array_to_base64(before_arr)
    result.after_image_b64 = image_array_to_base64(after_arr)

    # ------------------------------------------------------------------
    # 4. Change detection
    # ------------------------------------------------------------------
    trace.append("Change detection started (LAB perceptual differencing)")

    detector = ChangeDetector()
    detection = detector.detect(before_arr, after_arr)

    result.changed_area_percentage = round(
        detection.changed_area_fraction * 100, 2
    )
    trace.append(
        f"✓ Change detection complete: "
        f"{result.changed_area_percentage}% changed"
    )

    # ------------------------------------------------------------------
    # 5. Region extraction
    # ------------------------------------------------------------------
    trace.append("Region extraction started (connected component analysis)")

    regions = extract_regions(
        detection.change_mask,
        detection.magnitude_map,
    )
    result.changes = regions

    trace.append(
        f"✓ {len(regions)} significant change region(s) extracted"
    )

    # ------------------------------------------------------------------
    # 6. Visualisation
    # ------------------------------------------------------------------
    trace.append("Generating change map visualisation")

    result.change_map_b64 = generate_change_map(detection.magnitude_map)

    trace.append("✓ Change map generated")

    trace.append("Generating change overlay")

    result.change_overlay_b64 = generate_change_overlay(
        after_arr,
        detection.change_mask,
        detection.magnitude_map,
        regions,
    )

    trace.append("✓ Change overlay generated")

    # ------------------------------------------------------------------
    # 7. AI interpretation
    # ------------------------------------------------------------------
    trace.append("AI interpretation started")

    # Convert change map to bytes for VLM input
    from PIL import Image as PILImage
    import numpy as np

    heatmap_rgb = np.zeros((*detection.magnitude_map.shape, 3), dtype=np.float32)
    mag = np.clip(detection.magnitude_map, 0, 1) ** 0.7
    heatmap_rgb[..., 0] = np.clip(mag * 3.0, 0, 1)
    heatmap_rgb[..., 1] = np.clip(mag * 3.0 - 1.0, 0, 1)
    heatmap_rgb[..., 2] = np.clip(mag * 3.0 - 2.0, 0, 1)
    heatmap_img = PILImage.fromarray(
        (heatmap_rgb * 255).astype(np.uint8)
    )
    cm_buf = io.BytesIO()
    heatmap_img.save(cm_buf, format="PNG")
    change_map_bytes = cm_buf.getvalue()

    summary, confidence, is_simulated = interpret_changes(
        regions=regions,
        changed_area_pct=result.changed_area_percentage,
        query=query,
        before_image_bytes=before_bytes,
        after_image_bytes=after_bytes,
        change_map_bytes=change_map_bytes,
    )

    result.summary = summary
    result.overall_confidence = confidence
    result.simulated = is_simulated

    if is_simulated:
        trace.append("✓ AI interpretation complete (rule-based fallback)")
    else:
        trace.append("✓ AI interpretation complete (LLaVA VLM)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    trace.append(f"Pipeline complete in {elapsed:.2f}s")

    logger.info(
        "Change analysis pipeline complete: "
        "%d regions, %.1f%% changed, conf=%.2f, %.2fs",
        len(regions),
        result.changed_area_percentage,
        result.overall_confidence,
        elapsed,
    )

    return result
