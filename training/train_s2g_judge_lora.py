from __future__ import annotations

import argparse
import json
import os
from typing import Any

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = """You are a QA/RAG sufficiency judge.

Given a QUESTION and a CONTEXT (documents retrieved so far), decide whether the CONTEXT alone contains enough
information to reliably answer the QUESTION. If not, list the gap items that describe what information is still missing.

Output exactly one JSON object and nothing else.
The JSON object must have exactly two keys: "sufficient" and "gap_items".

Schema:
{
  "sufficient": true/false,
  "gap_items": [
    {
      "category": "bridge entity | attribute | relation | evidence span | other",
      "target": "string",
      "slot": "string",
      "description": "string"
    }
  ]
}

If "sufficient" is true, then "gap_items" must be [].
Use only the CONTEXT as evidence.
""".strip()


def extract_gap_items(example: dict[str, Any]) -> list[Any]:
    """Support both the new and older supervision field names."""
    for key in ("gap_items", "gap items", "missing_facts", "missing_info"):
        value = example.get(key)
        if isinstance(value, list):
            return value
    return []


def build_user_prompt(example: dict[str, Any]) -> str:
    """Format a training example as the user-side chat message."""
    question = str(example.get("question", "") or "").strip()
    context = str(example.get("context", "") or "").strip()
    return (
        "QUESTION:\n"
        f"{question}\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "Return only the JSON object."
    )


def build_assistant_output(example: dict[str, Any]) -> str:
    """Serialize the gold sufficiency label in the same schema used at inference time."""
    sufficient = bool(example.get("sufficient", False))
    label = {
        "sufficient": sufficient,
        "gap_items": [] if sufficient else extract_gap_items(example),
    }
    return json.dumps(label, ensure_ascii=False)


def format_example_to_text(example: dict[str, Any], tokenizer: AutoTokenizer) -> dict[str, str]:
    """Render a chat-formatted SFT training string."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example)},
        {"role": "assistant", "content": build_assistant_output(example)},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def build_lora_config() -> LoraConfig:
    """Use the LoRA target modules described in the paper."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
    """Create an SFT config while tolerating small TRL API differences."""
    base_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        dataset_text_field="text",
        packing=False,
    )

    candidate_kwargs = [
        {"evaluation_strategy": "steps", "eval_steps": args.eval_steps, "max_length": args.max_length},
        {"eval_strategy": "steps", "eval_steps": args.eval_steps, "max_length": args.max_length},
        {"evaluation_strategy": "steps", "eval_steps": args.eval_steps, "max_seq_length": args.max_length},
        {"eval_strategy": "steps", "eval_steps": args.eval_steps, "max_seq_length": args.max_length},
    ]

    last_error: TypeError | None = None
    for extra_kwargs in candidate_kwargs:
        try:
            return SFTConfig(**base_kwargs, **extra_kwargs)
        except TypeError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to build SFTConfig.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for the S2G-Judge model."
    )
    parser.add_argument("--data", required=True, help="Cleaned stage-2 supervision JSONL path.")
    parser.add_argument("--base-model", required=True, help="Base instruct model path or model id.")
    parser.add_argument("--output-dir", required=True, help="Training output directory.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--num-train-epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=2,
        help="Per-device train and eval batch size.",
    )
    parser.add_argument("--logging-steps", type=int, default=50, help="Logging interval.")
    parser.add_argument("--eval-steps", type=int, default=200, help="Evaluation interval.")
    parser.add_argument("--save-steps", type=int, default=200, help="Checkpoint save interval.")
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Number of checkpoints to keep.",
    )
    parser.add_argument("--fp16", action="store_true", help="Train with fp16.")
    parser.add_argument("--bf16", action="store_true", help="Train with bf16.")
    args = parser.parse_args()

    if not args.fp16 and not args.bf16:
        args.fp16 = True

    if args.fp16 and args.bf16:
        raise ValueError("Choose at most one mixed-precision mode.")

    dataset = load_dataset("json", data_files=args.data, split="train")
    if len(dataset) == 0:
        raise RuntimeError(f"No training examples were loaded from {args.data}")

    split = dataset.train_test_split(test_size=args.validation_ratio, seed=args.seed)
    train_raw = split["train"]
    eval_raw = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
    )
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    model = get_peft_model(base_model, build_lora_config())
    model.print_trainable_parameters()

    def apply_format(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        texts: list[str] = []
        batch_size = len(next(iter(batch.values())))
        for index in range(batch_size):
            example = {key: values[index] for key, values in batch.items()}
            texts.append(format_example_to_text(example, tokenizer)["text"])
        return {"text": texts}

    train_dataset = train_raw.map(
        apply_format,
        batched=True,
        remove_columns=train_raw.column_names,
        desc="Formatting training examples",
    )
    eval_dataset = eval_raw.map(
        apply_format,
        batched=True,
        remove_columns=eval_raw.column_names,
        desc="Formatting validation examples",
    )

    training_args = build_sft_config(args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_dir = os.path.join(args.output_dir, "lora")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"Loaded raw examples: {len(dataset)}")
    print(f"Training split size: {len(train_raw)}")
    print(f"Validation split size: {len(eval_raw)}")
    print(f"Saved LoRA adapter to: {adapter_dir}")


if __name__ == "__main__":
    main()
