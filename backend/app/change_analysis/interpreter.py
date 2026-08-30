"""
AI interpretation for bi-temporal change analysis.

Uses the existing LLaVA model (via app.models.vqa_model) when available.
Falls back to a rule-based interpreter that generates structured summaries
from the detected evidence.

Clearly separates DETECTED CHANGE (objective) from AI INTERPRETATION
(model-generated or rule-inferred).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from app.change_analysis.schemas import ChangeRegion

logger = logging.getLogger("satquery.change.interpreter")


def _build_evidence_summary(
    regions: List[ChangeRegion],
    changed_area_pct: float,
) -> str:
    """
    Build a structured textual summary of the objective detected evidence.
    This is fed to the AI model and also used as the fallback output.
    """
    if not regions:
        return (
            "No significant changes were detected between the two images. "
            "The scenes appear largely identical within the detection "
            "threshold."
        )

    lines = []
    lines.append(
        f"Detected {len(regions)} significant change region(s) covering "
        f"approximately {changed_area_pct:.1f}% of the total scene area."
    )

    for r in regions[:5]:  # summarise top 5
        lines.append(
            f"  • Region {r.region_id}: {r.location_description}, "
            f"area {r.area_fraction*100:.2f}%, "
            f"magnitude {r.mean_magnitude:.2f}, "
            f"confidence {r.confidence*100:.0f}%"
        )

    return "\n".join(lines)


def _rule_based_interpretation(
    regions: List[ChangeRegion],
    changed_area_pct: float,
    query: str = "",
) -> Tuple[str, float]:
    """
    Generate a structured interpretation without a VLM.

    Returns (summary_text, confidence).
    """
    if not regions:
        return (
            "No significant changes were detected between the before and "
            "after images. The two scenes appear largely consistent.",
            0.90,
        )

    q = query.lower()

    # Analyse dominant change locations
    largest = regions[0]
    locations = list({r.location_description for r in regions[:3]})
    loc_str = ", ".join(locations)

    # Confidence based on region count and magnitude
    avg_conf = sum(r.confidence for r in regions) / len(regions)

    summary_parts = []

    summary_parts.append(
        f"Significant changes were detected between the two temporal "
        f"observations. {len(regions)} distinct change region(s) were "
        f"identified, covering approximately {changed_area_pct:.1f}% "
        f"of the scene."
    )

    summary_parts.append(
        f"The most prominent changes are located in the {loc_str}."
    )

    if largest.mean_magnitude > 0.6:
        summary_parts.append(
            "The magnitude of the dominant change region is high, suggesting "
            "substantial land-cover transformation (e.g. new construction, "
            "deforestation, or flooding)."
        )
    elif largest.mean_magnitude > 0.3:
        summary_parts.append(
            "The change magnitude is moderate, which may indicate gradual "
            "land-use transition, seasonal vegetation change, or partial "
            "development."
        )
    else:
        summary_parts.append(
            "The change magnitude is relatively low, suggesting subtle "
            "changes such as minor vegetation variation or atmospheric "
            "differences between acquisitions."
        )

    # Query-specific additions
    if any(kw in q for kw in ["built", "urban", "construct", "building"]):
        summary_parts.append(
            "Based on the query context, the detected changes may correspond "
            "to built-up area expansion. Ground truth validation is "
            "recommended for confirmation."
        )
    elif any(kw in q for kw in ["veget", "green", "forest", "tree", "ndvi"]):
        summary_parts.append(
            "Based on the query context, the changes may relate to "
            "vegetation cover dynamics (gain or loss). Spectral indices "
            "like NDVI would provide more precise classification."
        )
    elif any(kw in q for kw in ["water", "flood", "river", "lake"]):
        summary_parts.append(
            "Based on the query context, the changes may involve water "
            "body extent variation. Water index analysis (NDWI/MNDWI) "
            "would improve classification accuracy."
        )

    return " ".join(summary_parts), round(avg_conf, 3)


def interpret_changes(
    regions: List[ChangeRegion],
    changed_area_pct: float,
    query: str = "",
    before_image_bytes: Optional[bytes] = None,
    after_image_bytes: Optional[bytes] = None,
    change_map_bytes: Optional[bytes] = None,
) -> Tuple[str, float, bool]:
    """
    Generate an AI interpretation of the detected changes.

    Tries the real LLaVA model first; falls back to rule-based.

    Returns
    -------
    (summary_text, confidence, is_simulated)
    """

    evidence = _build_evidence_summary(regions, changed_area_pct)

    # --- Try the real VLM ---
    try:
        from app.models import vqa_model

        if vqa_model.is_available() and change_map_bytes:
            prompt_query = (
                f"You are a remote-sensing change analyst. "
                f"Two satellite images of the same region were compared. "
                f"Here is the detected evidence:\n{evidence}\n\n"
                f"User question: {query or 'What changed between these images?'}\n\n"
                f"Provide a concise explanation of what changed, where, "
                f"and the likely category (built-up, vegetation, water, etc.)."
            )

            answer, conf = vqa_model.answer_vqa(change_map_bytes, prompt_query)

            logger.info("VLM interpretation generated (conf=%.2f)", conf)

            # Combine evidence and VLM answer
            full_summary = (
                f"DETECTED CHANGE:\n{evidence}\n\n"
                f"AI INTERPRETATION:\n{answer}"
            )

            return full_summary, conf, False

    except Exception as e:
        logger.warning("VLM interpretation failed: %s — using rule-based", e)

    # --- Fallback: rule-based ---
    summary, conf = _rule_based_interpretation(regions, changed_area_pct, query)

    full_summary = (
        f"DETECTED CHANGE:\n{evidence}\n\n"
        f"AI INTERPRETATION:\n{summary}"
    )

    return full_summary, conf, True
