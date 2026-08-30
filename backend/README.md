<<<<<<< HEAD
# SatQuery AI — Backend Service

SatQuery AI is a multimodal satellite intelligence and Earth observation query system providing three core analysis modes:
1. **Single-Image Visual Question Answering (VQA)** (`POST /api/analyze`)
2. **Bi-Temporal Change Analysis** (`POST /api/analyze/change`)
3. **Optical + SAR Multimodal Fusion** (`POST /api/analyze/fusion`)

---

## Capabilities & Architecture

```
app/
  main.py                     FastAPI application, health and analysis endpoints
  config.py                   Environment configuration and model switches
  schemas.py                  Pydantic models for single-image, bi-temporal, and fusion responses
  models/
    vqa_model.py              LLaVA-1.5 multimodal VLM + optional fine-tuned LoRA
  agent/
    router.py                 Task classification and dynamic orchestration
  change_analysis/            Bi-temporal change detection pipeline
    preprocessing.py          Radiometric and geometric normalization
    detector.py               Spectral distance and change masking
    region_extractor.py       Bounding box extraction and change ranking
    pipeline.py               Orchestrator and visual product generator
  fusion/                     Optical + SAR Multimodal Fusion engine
    preprocessing.py          Optical normalization and SAR despeckling/dB scaling
    alignment.py              Geospatial CRS / dimension grid matching
    feature_extraction.py     Optical spectral indices & SAR backscatter structures
    fusion_engine.py          Cross-modal agreement heatmap & composite generation
    reasoning.py              Query-aware multimodal evidence synthesis
    visualization.py          Base64 PNG visual products (RGB, dB, Composite, Heatmap)
    pipeline.py               End-to-end 8-stage fusion pipeline
```

---

## Analysis Endpoints

### 1. Optical + SAR Multimodal Fusion (`POST /api/analyze/fusion`)

Combines multispectral optical imagery with synthetic aperture radar (SAR) backscatter for all-weather scene interpretation and feature identification.

- **Request Parameters** (`multipart/form-data`):
  - `query` (str, optional): Natural-language question or prompt.
  - `optical_image` (file, required): Multispectral or RGB optical satellite scene (PNG, JPEG, TIFF).
  - `sar_image` (file, required): SAR radar amplitude or backscatter scene (PNG, JPEG, TIFF).
  - `fusion_method` (str, optional): `"composite"` (False-color Optical-SAR composite) or `"ihs"` (IHS spatial texture modulation).

- **Response Schema (`FusionAnalysisResponse`)**:
  - `task`: `"optical_sar_fusion"`
  - `query`: Query submitted by user
  - `summary`: High-level multimodal executive summary
  - `optical_evidence`: Dedicated optical observations (spectral response, vegetation, cloud shadows)
  - `sar_evidence`: Dedicated SAR radar observations (dB backscatter, double-bounce, specular water)
  - `fused_interpretation`: Cross-modal consensus, disambiguation, and query answer
  - `confidence`: Confidence score (0–100%)
  - `modality_agreement_percentage`: Percentage of sensor agreement across the scene
  - `optical_image`: Base64 PNG preprocessed optical preview
  - `sar_image`: Base64 PNG log-scaled radar backscatter preview
  - `fusion_visualization`: Base64 PNG fused composite with localized feature bounding boxes
  - `evidence_map`: Base64 PNG cross-modal agreement heatmap
  - `features`: List of localized cross-modal features with coordinates, categories, and evidence
  - `alignment_method`: `"geo_referenced"` or `"dimension_matched"`
  - `alignment_details`: Explanation of spatial alignment applied
  - `execution_trace`: Step-by-step pipeline execution trace
  - `simulated`: Boolean indicating if rule-based fallback was utilized

---

### 2. Bi-Temporal Change Analysis (`POST /api/analyze/change`)

Detects, quantifies, and isolates structural and environmental changes between two co-registered satellite observations.

- **Request Parameters** (`multipart/form-data`):
  - `query` (str, optional): Description or question regarding expected change.
  - `before_image` (file, required): Pre-event observation.
  - `after_image` (file, required): Post-event observation.

---

### 3. Single-Image VQA (`POST /api/analyze`)

Executes Visual Question Answering on single satellite scenes using LLaVA-1.5 VLM or agent router dispatching.

---

## Running the Backend

### Local / CPU Mode (Rule-based & Fallback)

```bash
cd backend
pip install fastapi uvicorn python-multipart pydantic pillow scipy httpx
uvicorn app.main:app --reload --port 8000
```

Verify service status:
```bash
curl http://localhost:8000/health
```

### Running with GPU Acceleration & LLaVA VLM
=======
# SatQuery AI — Backend (Phase 1: Single-Image VQA)

This is the first real backend slice for SatQuery AI: a FastAPI service that
runs single-image VQA through LLaVA-1.5 (+ an optional LoRA adapter you
fine-tune on remote-sensing data), wired to return the exact JSON shape your
existing frontend (`script.js`) already expects.

## Structure

```
app/
  main.py            FastAPI app, /health and /api/analyze endpoints
  config.py           model paths / device / on-off switches (env vars)
  schemas.py           response shapes matching the frontend's DEMOS contract
  models/
    vqa_model.py       loads LLaVA-1.5 + optional LoRA, runs inference
  agent/
    router.py          task classification, input validation, orchestration
training/
  prepare_rsvqa_data.py   convert RSVQA annotations -> unified JSONL
  train_lora_vqa.py       LoRA fine-tuning script (needs GPU)
```

## Why it's split this way

- `agent/router.py` is the "agentic controller" the problem statement asks
  for: it classifies the query, validates inputs, picks a specialist, and
  returns an auditable trace. Right now only `vqa` is wired to a real model;
  `grounding`/`change`/`fusion`/`captioning` return clearly-labeled simulated
  stubs so you can build those next using the exact same pattern.
- `models/vqa_model.py` degrades gracefully: if torch/transformers/peft
  aren't installed, or no GPU is available, `is_available()` returns False
  and the router automatically falls back to a simulated answer instead of
  crashing the whole API. This means you can run/demo the backend on a
  laptop with no GPU while you separately train the real model elsewhere.

## Running the backend (no GPU / just wiring the API)

```bash
cd satquery-backend
python -m venv venv && source venv/bin/activate
pip install fastapi uvicorn python-multipart pydantic pillow
export SATQUERY_LOAD_MODEL=0        # skip trying to load the 7B model
uvicorn app.main:app --reload --port 8000
```

Visit http://localhost:8000/health — you should see
`"vqa_model_loaded": false"`, and `/api/analyze` will return simulated
answers. This confirms the plumbing works.

## Running with the real model (GPU machine, e.g. Colab/Kaggle/your own box)
>>>>>>> 0b0e83f4bd5cc0f3a71cf5766ab1e38cc5215a9a

```bash
pip install -r requirements.txt
export SATQUERY_LOAD_MODEL=1
export SATQUERY_DEVICE=cuda
<<<<<<< HEAD
export SATQUERY_LORA_ADAPTER_PATH=./lora-rsvqa
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Running Automated Tests

```bash
cd backend
python test_fusion.py
```

This runs the complete 10-test suite verifying:
- Single-Image VQA endpoint
- Bi-Temporal Change Detection endpoint
- Optical + SAR Fusion endpoint
- Missing optical / SAR payload validation
- Corrupt image handling
- Mismatched dimension auto-alignment
- Spatial metadata extraction
- Agent router general endpoint dispatching
=======
export SATQUERY_LORA_ADAPTER_PATH=./lora-rsvqa   # after you've trained it, see below
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Training the LoRA adapter on RSVQA

1. Download the RSVQA (and/or BigEarthNet-text) dataset per the problem
   statement's dataset links.
2. Convert it to the unified format:
   ```bash
   python training/prepare_rsvqa_data.py \
       --rsvqa_json RSVQA_LR_split_train_questions.json \
       --rsvqa_answers RSVQA_LR_split_train_answers.json \
       --image_dir ./rsvqa_images \
       --out train_vqa.jsonl
   ```
   Print a few parsed rows and sanity-check them against the raw files first
   — RSVQA's JSON field names differ slightly between LR/HR/HRv2 releases,
   so `parse_rsvqa_json` may need small tweaks to match what you downloaded.
3. Fine-tune:
   ```bash
   python training/train_lora_vqa.py \
       --train_file train_vqa.jsonl \
       --output_dir ./lora-rsvqa \
       --epochs 3 --batch_size 4
   ```
   On a single T4/A10 this will take a while for a full RSVQA split — start
   with a small subset (a few hundred examples) to confirm the loop runs
   end-to-end before committing to a full run.
4. Point the backend at the result via `SATQUERY_LORA_ADAPTER_PATH`.

## Wiring the existing frontend to this backend

Your current `script.js` fakes everything client-side in `runAnalysis()` /
`finishAnalysisRun()`. The smallest real change is: after your existing
execution-trace animation finishes, call the real backend instead of just
reading from the local `DEMOS` object, for the `vqa` task specifically (keep
the other tasks on the simulated path until their models exist too).

```js
async function callBackendVQA(query, imageFile) {
  const form = new FormData();
  form.append('query', query);
  form.append('image1', imageFile);
  const resp = await fetch('http://localhost:8000/api/analyze', {
    method: 'POST',
    body: form
  });
  if (!resp.ok) throw new Error('backend error');
  return resp.json(); // { task, model, answer, confidence, trace, simulated }
}
```

Then in `finishAnalysisRun(demo)`, for `selectedDemo === 'vqa'`, call
`callBackendVQA(query, uploadedAssets.img1.file)` and use its `answer` /
`confidence` / `model` fields in place of the hardcoded `demo` object —
same rendering code, real data underneath.

## What's next (not yet built here)

- Region grounding, change VQA, and optical-SAR fusion models, following the
  exact same pattern as `vqa_model.py` + `router.run_vqa()`.
- Swapping the keyword-based `classify_task()` for something more robust if
  keyword matching proves too brittle on real queries.
- Confidence calibration — the current confidence is a rough token-probability
  proxy, not a calibrated accuracy estimate.
>>>>>>> 0b0e83f4bd5cc0f3a71cf5766ab1e38cc5215a9a
