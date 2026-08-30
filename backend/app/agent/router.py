"""
Minimal agentic controller.

Responsibilities (matches the problem statement's "Agentic Model and Tool
Orchestration" section):
  1. interpret the query and classify the requested task
  2. check number/modality/format of input images
  3. select the appropriate specialist tool from the registry
  4. execute it and collect outputs
  5. return an auditable execution trace (task, tool, key params, outputs)

Only single-image VQA is wired to a real model right now (see
app/models/vqa_model.py). The other tasks (grounding, change, fusion) are
stubbed with clearly-labeled simulated outputs so the API contract and
frontend integration are already correct -- swap in real specialist models
into TOOL_REGISTRY as you build them, following the same pattern as
`run_vqa`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from app.models import vqa_model
from app.schemas import AnalyzeResponse


@dataclass
class ImageInput:
    filename: str
    content_type: str
    data: bytes


# ---------------------------------------------------------------------------
# 1. Query intent classification
# ---------------------------------------------------------------------------
# Keyword-based for now -- deliberately simple and auditable. This is the
# piece most worth upgrading to a small instruction-tuned LLM call later
# (the routing logic itself doesn't need RS adaptation, only the specialist
# models it calls do).
def classify_task(query: str, num_images: int) -> str:
    """
    Classifies intent from the query text alone. Deliberately does NOT
    factor in num_images here -- a query like "what changed between these
    two images?" is still a *change* request even if the person forgot to
    upload the second image. validate_inputs() is responsible for catching
    that mismatch and returning a clear error, rather than this function
    silently downgrading the task to VQA.
    """
    q = query.lower()
    if any(w in q for w in ["chang", "differ", "compar", "between these"]):
        return "change"
    if any(w in q for w in ["sar", "fus", "radar", "optical and"]):
        return "fusion"
    if any(w in q for w in ["highlight", "locate", "where is", "point to", "find the"]):
        return "grounding"
    if any(w in q for w in ["caption", "describe", "summari"]):
        return "captioning"
    return "vqa"


# ---------------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------------
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/tiff", "image/tif"}


def validate_inputs(task: str, images: List[ImageInput]) -> Optional[str]:
    """Returns an error string if invalid, else None."""
    if not images:
        return "At least one image is required."
    for img in images:
        if img.content_type not in ALLOWED_TYPES:
            return f"Unsupported format '{img.content_type}' for {img.filename}."
    if task in ("change", "fusion") and len(images) < 2:
        return f"Task '{task}' requires two co-registered images (received {len(images)})."
    return None


# ---------------------------------------------------------------------------
# 3/4. Specialist execution
# ---------------------------------------------------------------------------
def run_vqa(query: str, images: List[ImageInput], trace: List[str]) -> AnalyzeResponse:
    trace.append("Task identified: Single-Image Visual Question Answering")
    if vqa_model.is_available():
        trace.append(
            "Specialist model selected: LLaVA-1.5"
            + (" + LoRA (remote-sensing adapted)" if vqa_model.using_lora() else " (base, NOT RS-adapted)")
        )
        trace.append("Running inference on uploaded image")
        answer, conf = vqa_model.answer_vqa(images[0].data, query)
        trace.append("Answer generated")
        return AnalyzeResponse(
            task="Single-Image VQA",
            model="LLaVA-1.5" + ("-LoRA-RS" if vqa_model.using_lora() else "-base"),
            query=query,
            answer=answer,
            confidence=int(round(conf * 100)),
            trace=trace,
            simulated=False,
        )
    else:
        trace.append(f"Real VQA model unavailable ({vqa_model.load_error()}) -- using simulated fallback")
        return AnalyzeResponse(
            task="Single-Image VQA",
            model="Simulated (no model loaded)",
            query=query,
            answer="Several agricultural fields, roads, built-up structures and a "
                   "water body are visible in this scene. (SIMULATED -- connect a "
                   "loaded VQA model for a real answer.)",
            confidence=60,
            trace=trace,
            simulated=True,
        )


def run_stub(task_name: str, model_name: str, query: str, trace: List[str]) -> AnalyzeResponse:
    """Placeholder for grounding / change / fusion / captioning until those
    specialist models are trained and wired in the same way run_vqa() is."""
    trace.append(f"Task identified: {task_name}")
    trace.append(f"Specialist model '{model_name}' not yet integrated -- returning simulated output")
    return AnalyzeResponse(
        task=task_name,
        model=f"Simulated ({model_name} not yet trained/integrated)",
        query=query,
        answer=(
            f"This is a placeholder response for '{task_name}'. Train and wire in "
            f"the {model_name} following the same pattern as app/models/vqa_model.py "
            f"and app.agent.router.run_vqa()."
        ),
        confidence=50,
        trace=trace,
        simulated=True,
    )


TASK_DISPLAY_NAMES = {
    "vqa": "Single-Image VQA",
    "grounding": "Region Grounding",
    "change": "Bi-Temporal Change Analysis",
    "fusion": "Optical + SAR Fusion",
    "captioning": "Scene Captioning",
}
TASK_MODEL_NAMES = {
    "grounding": "Region Grounding Model",
    "change": "Change Understanding Model",
    "fusion": "Optical + SAR Fusion Model",
    "captioning": "RS Captioning Model",
}


def run_agent(query: str, images: List[ImageInput]) -> AnalyzeResponse:
    trace: List[str] = ["Query received", "Query classified"]

    task = classify_task(query, len(images))
    error = validate_inputs(task, images)
    if error:
        trace.append(f"Input validation FAILED: {error}")
        return AnalyzeResponse(
            task="Validation Error",
            model="-",
            query=query,
            answer=error,
            confidence=0,
            trace=trace,
            simulated=True,
        )
    trace.append(f"Input validated: {len(images)} image(s)")

    if task == "vqa":
        return run_vqa(query, images, trace)

    # --- Bi-temporal change analysis (real pipeline) ---
    if task == "change" and len(images) >= 2:
        try:
            trace.append("Task identified: Bi-Temporal Change Analysis")
            trace.append("Routing to change analysis pipeline")

            from app.change_analysis.pipeline import run_change_analysis

            result = run_change_analysis(
                before_bytes=images[0].data,
                after_bytes=images[1].data,
                query=query,
            )

            # Merge pipeline trace into agent trace
            trace.extend(result.execution_trace)

            # Build summary answer text for the existing response format
            answer = result.summary or "Change analysis completed."
            conf = int(round(result.overall_confidence * 100))

            trace.append("Change analysis pipeline complete")

            return AnalyzeResponse(
                task="Bi-Temporal Change Analysis",
                model="Change Detection Pipeline"
                      + (" + LLaVA" if not result.simulated else " (rule-based)"),
                query=query,
                answer=answer,
                confidence=conf,
                trace=trace,
                visual_evidence=result.change_map_b64 or None,
                simulated=result.simulated,
            )

        except Exception as e:
            trace.append(f"Change pipeline error: {e} — falling back to stub")

    # --- Optical + SAR Multimodal Fusion (real pipeline) ---
    if task == "fusion" and len(images) >= 2:
        try:
            trace.append("Task identified: Optical + SAR Multimodal Fusion")
            trace.append("Routing to Optical + SAR fusion pipeline")

            from app.fusion.pipeline import run_fusion_analysis

            result = run_fusion_analysis(
                optical_bytes=images[0].data,
                sar_bytes=images[1].data,
                query=query,
            )

            trace.extend(result.execution_trace)
            answer = result.summary or result.fused_interpretation or "Fusion analysis completed."
            conf = result.confidence

            trace.append("Fusion pipeline complete")

            return AnalyzeResponse(
                task="Optical + SAR Fusion",
                model="Multimodal Fusion Pipeline"
                      + (" + LLaVA" if not result.simulated else " (rule-based)"),
                query=query,
                answer=answer,
                confidence=conf,
                trace=trace,
                visual_evidence=result.fusion_visualization_b64 or None,
                simulated=result.simulated,
            )

        except Exception as e:
            trace.append(f"Fusion pipeline error: {e} — falling back to stub")

    return run_stub(TASK_DISPLAY_NAMES[task], TASK_MODEL_NAMES[task], query, trace)
