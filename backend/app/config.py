import os

# Base LLaVA checkpoint (downloaded from Hugging Face Hub the first time it runs,
# so the machine that runs this needs internet access to huggingface.co).
BASE_MODEL_ID = os.environ.get("SATQUERY_BASE_MODEL", "llava-hf/llava-1.5-7b-hf")

# Path to your fine-tuned LoRA adapter directory, produced by
# training/train_lora_vqa.py. Leave empty/unset to run the base model with
# no RS adaptation (fine for wiring/testing, NOT for the final submission --
# the problem statement requires RS fine-tuning).
LORA_ADAPTER_PATH = os.environ.get("SATQUERY_LORA_ADAPTER_PATH", "")

# "cuda", "mps", or "cpu". CPU inference on a 7B VLM will be very slow --
# only use it to confirm the pipeline runs end-to-end.
DEVICE = os.environ.get("SATQUERY_DEVICE", "cuda")

# Whether to actually try to load the model at startup. Set to "0" while you
# are just wiring up the frontend/backend contract and don't have a GPU handy --
# the API will then always return the simulated fallback path.
LOAD_MODEL_ON_STARTUP = os.environ.get("SATQUERY_LOAD_MODEL", "1") == "1"

MAX_NEW_TOKENS = int(os.environ.get("SATQUERY_MAX_NEW_TOKENS", "128"))
