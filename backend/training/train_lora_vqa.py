"""
LoRA fine-tuning of LLaVA-1.5 on remote-sensing VQA data.

Run this on a GPU machine (Colab/Kaggle T4 or better, or your own GPU box)
with internet access to the Hugging Face Hub -- it downloads the base
llava-hf/llava-1.5-7b-hf checkpoint the first time.

Input: a JSONL file produced by prepare_rsvqa_data.py, one example per line:
    {"image": "path/to/img.png", "question": "...", "answer": "..."}

Usage:
    python train_lora_vqa.py \
        --train_file train_vqa.jsonl \
        --output_dir ./lora-rsvqa \
        --epochs 3 --batch_size 4 --lr 2e-4

After training, point the backend at the adapter:
    export SATQUERY_LORA_ADAPTER_PATH=./lora-rsvqa
"""
import argparse
import json

import torch
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(train_file: str, processor):
    rows = load_jsonl(train_file)
    ds = Dataset.from_list(rows)

    def preprocess(example):
        image = Image.open(example["image"]).convert("RGB")
        prompt = f"USER: <image>\n{example['question']}\nASSISTANT: {example['answer']}"
        enc = processor(text=prompt, images=image, return_tensors="pt", padding="max_length",
                         truncation=True, max_length=256)
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        return enc

    # NOTE: for a real training run, do this preprocessing lazily with a
    # torch Dataset/DataLoader instead of .map(), since holding every image
    # tensor in memory at once for a large RSVQA split will not fit in RAM.
    # Kept as .map() here for clarity/readability of the reference pipeline.
    return ds.map(preprocess, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    print(f"Loading base model: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        # target the language-model attention projections; adjust if your
        # transformers version names these layers differently
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Building dataset...")
    train_ds = build_dataset(args.train_file, processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)
    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter + processor to {args.output_dir}")


if __name__ == "__main__":
    main()
