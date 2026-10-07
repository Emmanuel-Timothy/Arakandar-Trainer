"""Train a reviewed Arakandar conversation adapter on the local 3B model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            if not str(item.get("question", "")).strip():
                raise ValueError(f"Line {line_number} is missing question")
            if not str(item.get("answer", "")).strip():
                raise ValueError(f"Line {line_number} is missing answer")
            if not isinstance(item.get("evidence", {}), dict):
                raise ValueError(f"Line {line_number} evidence must be an object")
            examples.append(item)
    if len(examples) < 8:
        raise ValueError("At least 8 reviewed examples are required for a training run")
    return examples


def format_example(example: dict[str, Any]) -> list[dict[str, str]]:
    evidence = json.dumps(example["evidence"], ensure_ascii=True, sort_keys=True)
    return [
        {
            "role": "system",
            "content": (
                "You are Arakandar, a grounded market research assistant. Use only the "
                "supplied evidence. Never invent prices, news, probabilities, or trades. "
                "The deterministic signal is authoritative. This is not financial advice."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {example['question']}\nEvidence: {evidence}",
        },
        {"role": "assistant", "content": str(example["answer"]).strip()},
    ]


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training requires the optional LoRA packages; install requirements-local-llm.txt"
        ) from exc

    examples = load_examples(args.dataset)
    split_at = max(1, round(len(examples) * 0.8))
    if split_at >= len(examples):
        split_at = len(examples) - 1
    train_examples = examples[:split_at]
    eval_examples = examples[split_at:]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    def tokenize(item: dict[str, Any]) -> dict[str, list[int]]:
        prompt = tokenizer.apply_chat_template(format_example(item), tokenize=False)
        return tokenizer(prompt, truncation=True, max_length=args.max_length)

    train_tokens = [tokenize(item) for item in train_examples]
    eval_tokens = [tokenize(item) for item in eval_examples]
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    output_dir = args.output
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            learning_rate=2e-4,
            warmup_ratio=0.05,
            logging_steps=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False,
        ),
        train_dataset=train_tokens,
        eval_dataset=eval_tokens,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "base_model": str(args.base_model),
                "examples": len(examples),
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "method": "LoRA",
                "review_required": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a reviewed Arakandar 3B LoRA adapter")
    parser.add_argument("--base-model", type=Path, default=Path("models/base/arakandar-3b"))
    parser.add_argument("--dataset", type=Path, default=Path("data/llm/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/adapters/arakandar-3b-lora"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if not args.base_model.is_dir():
        parser.error(f"Base model directory does not exist: {args.base_model}")
    if not args.dataset.is_file():
        parser.error(f"Reviewed dataset does not exist: {args.dataset}")
    try:
        train(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
