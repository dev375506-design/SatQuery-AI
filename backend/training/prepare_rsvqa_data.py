"""
Converts RSVQA-style annotation files (and, optionally, BigEarthNet image-text
pairs) into a single unified JSONL file for LoRA fine-tuning:

    {"image": "path/to/image.png", "question": "...", "answer": "..."}

RSVQA's official release ships question/answer pairs alongside an
image-id -> file mapping in JSON; the exact field names vary slightly
between the LR/HR/HRv2 releases, so adjust `parse_rsvqa_json` to match
whichever split you download. This script is meant as a starting point,
not a drop-in for every dataset revision -- always print a few parsed
examples and sanity-check them against the raw files before training on
the output.

Usage:
    python prepare_rsvqa_data.py \
        --rsvqa_json path/to/RSVQA_LR_split_train_questions.json \
        --rsvqa_answers path/to/RSVQA_LR_split_train_answers.json \
        --image_dir path/to/rsvqa/images \
        --out train_vqa.jsonl
"""
import argparse
import json
import os


def parse_rsvqa_json(questions_path: str, answers_path: str, image_dir: str):
    with open(questions_path) as f:
        questions = json.load(f)
    with open(answers_path) as f:
        answers = json.load(f)

    # Build an answer lookup by question id -- adjust key names to match your
    # actual downloaded file structure; RSVQA releases nest these under
    # different top-level keys depending on version.
    answer_by_qid = {a["question_id"]: a["answer"] for a in answers.get("answers", answers)}

    examples = []
    for q in questions.get("questions", questions):
        qid = q["id"]
        img_id = q["img_id"]
        question_text = q["question"]
        answer_text = answer_by_qid.get(qid)
        if answer_text is None:
            continue
        image_path = os.path.join(image_dir, f"{img_id}.tif")
        if not os.path.exists(image_path):
            # try common alternative extensions
            for ext in (".png", ".jpg", ".jpeg"):
                alt = os.path.join(image_dir, f"{img_id}{ext}")
                if os.path.exists(alt):
                    image_path = alt
                    break
        examples.append({"image": image_path, "question": question_text, "answer": str(answer_text)})
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsvqa_json", required=True)
    ap.add_argument("--rsvqa_answers", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    examples = parse_rsvqa_json(args.rsvqa_json, args.rsvqa_answers, args.image_dir)
    print(f"Parsed {len(examples)} examples. First 3:")
    for e in examples[:3]:
        print(" ", e)

    with open(args.out, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
