"""
SatQuery AI - LLaVA VQA model loader.

Uses 4-bit quantization so LLaVA-1.5-7B can be attempted
on GPUs with limited VRAM such as an RTX 3050 6GB.

Supports an optional PEFT/LoRA adapter through:
    SATQUERY_LORA_ADAPTER_PATH
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

from PIL import Image

from app import config

logger = logging.getLogger("satquery.vqa_model")

_model = None
_processor = None
_load_error: Optional[str] = None
_using_lora = False


def _try_load():
    """Load LLaVA once. Fail gracefully to simulated mode."""
    global _model, _processor, _load_error, _using_lora

    if _model is not None or _load_error is not None:
        return

    if not config.LOAD_MODEL_ON_STARTUP:
        _load_error = "Model loading disabled via SATQUERY_LOAD_MODEL=0"
        return

    try:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            LlavaForConditionalGeneration,
        )

        # -------------------------------------------------
        # Check CUDA
        # -------------------------------------------------

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. "
                "A CUDA-enabled PyTorch installation is required "
                "for LLaVA inference."
            )

        device = config.DEVICE

        if device != "cuda":
            logger.warning(
                "Configured device is '%s'. "
                "Using CUDA because bitsandbytes quantization is intended "
                "for the NVIDIA GPU.",
                device,
            )
            device = "cuda"

        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        logger.info("Loading base model: %s", config.BASE_MODEL_ID)

        # -------------------------------------------------
        # Processor (do NOT pass use_fast — LlavaProcessor
        # wraps an AutoImageProcessor which has no slow variant)
        # -------------------------------------------------

        _processor = AutoProcessor.from_pretrained(
            config.BASE_MODEL_ID,
        )

        # -------------------------------------------------
        # Model — try 4-bit first, fall back to float16
        # -------------------------------------------------

        model = None

        # Attempt 1: 4-bit quantization via bitsandbytes
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

            model = LlavaForConditionalGeneration.from_pretrained(
                config.BASE_MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            logger.info("Model loaded with 4-bit quantization.")

        except Exception as bnb_err:
            logger.warning(
                "4-bit quantization failed (%s). "
                "Attempting float16 loading instead. "
                "This requires more VRAM (~14 GB for 7B).",
                bnb_err,
            )

            # Attempt 2: plain float16 (no bitsandbytes)
            model = LlavaForConditionalGeneration.from_pretrained(
                config.BASE_MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            logger.info("Model loaded in float16 (no quantization).")

        # -------------------------------------------------
        # Optional LoRA adapter
        # -------------------------------------------------

        if config.LORA_ADAPTER_PATH:
            from peft import PeftModel

            logger.info(
                "Attaching LoRA adapter from: %s",
                config.LORA_ADAPTER_PATH,
            )

            model = PeftModel.from_pretrained(
                model,
                config.LORA_ADAPTER_PATH,
            )

            _using_lora = True

            logger.info("LoRA adapter successfully attached.")

        else:
            logger.warning(
                "No LORA_ADAPTER_PATH set. "
                "Running the base LLaVA model without "
                "remote-sensing LoRA adaptation."
            )

        # -------------------------------------------------
        # Evaluation mode
        # -------------------------------------------------

        model.eval()

        _model = model

        logger.info(
            "VQA model ready. device=%s, lora=%s",
            device,
            _using_lora,
        )

    except Exception as e:
        logger.exception(
            "Failed to load VQA model. "
            "Falling back to simulated mode."
        )

        _load_error = str(e)



def is_available() -> bool:
    """Return True if the real VQA model loaded successfully."""
    _try_load()
    return _model is not None


def using_lora() -> bool:
    """Return True when a LoRA adapter is attached."""
    return _using_lora


def load_error() -> Optional[str]:
    """Return the model-loading error, if any."""
    return _load_error


def answer_vqa(
    image_bytes: bytes,
    question: str,
) -> Tuple[str, float]:
    """
    Run single-image visual question answering.

    Returns:
        (answer_text, confidence_0_to_1)
    """

    _try_load()

    if _model is None:
        raise RuntimeError(
            f"VQA model not available: {_load_error}"
        )

    import torch

    # -------------------------------------------------
    # Load image
    # -------------------------------------------------

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # -------------------------------------------------
    # LLaVA prompt
    # -------------------------------------------------

    prompt = (
        "USER: <image>\n"
        "You are a remote-sensing image analyst. "
        "Analyze the satellite image carefully and answer "
        "the user's question concisely and factually.\n"
        f"Question: {question}\n"
        "ASSISTANT:"
    )

    # -------------------------------------------------
    # Processor
    # -------------------------------------------------

    inputs = _processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    # Move tensors to the model's device.
    inputs = {
        key: value.to("cuda") if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    # -------------------------------------------------
    # Generate
    # -------------------------------------------------

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # -------------------------------------------------
    # Decode
    # -------------------------------------------------

    input_length = inputs["input_ids"].shape[1]

    generated_ids = output.sequences[
        0,
        input_length:,
    ]

    text = _processor.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    # -------------------------------------------------
    # Rough confidence estimate
    # -------------------------------------------------

    try:
        scores = output.scores

        if scores:
            probabilities = [
                torch.softmax(
                    score[0],
                    dim=-1,
                ).max().item()
                for score in scores
            ]

            confidence = (
                sum(probabilities)
                / len(probabilities)
            )
        else:
            confidence = 0.75

    except Exception:
        confidence = 0.75

    return text, float(confidence)