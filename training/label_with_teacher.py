from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


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

Constraints:
- Use ONLY the CONTEXT as evidence.
- Do not rely on parametric knowledge or external assumptions.
- If "sufficient" is true, then "gap_items" must be [].
- If "sufficient" is false, "gap_items" should contain the missing links needed to answer the question.
""".strip()


def build_user_prompt(example: dict[str, Any]) -> str:
    """Format a single `(question, context)` snapshot for teacher labeling."""
    question = str(example.get("question", "") or "").strip()
    context = str(example.get("context", "") or "").strip()
    return (
        "QUESTION:\n"
        f"{question}\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "Return only the JSON object."
    )


def extract_gap_items(payload: dict[str, Any]) -> list[Any]:
    """Support both the new and older gap-field spellings during parsing."""
    for key in ("gap_items", "gap items", "missing_facts", "missing_info"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def normalize_teacher_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize teacher output to the canonical schema."""
    sufficient = payload.get("sufficient")
    if not isinstance(sufficient, bool):
        return None

    gap_items = extract_gap_items(payload)
    if sufficient:
        gap_items = []

    return {
        "sufficient": sufficient,
        "gap_items": gap_items,
    }


def load_processed_keys(output_path: Path) -> set[tuple[Any, Any]]:
    """Collect already-labeled `(id, turn)` keys for resumable runs."""
    if not output_path.exists():
        return set()

    processed: set[tuple[Any, Any]] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            processed.add((record.get("id"), record.get("turn")))
    return processed


def call_teacher(
    client: OpenAI,
    model: str,
    user_prompt: str,
    max_tokens: int,
    max_retries: int,
    request_timeout: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """Call the teacher model and return both the parsed payload and raw text."""
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=max_tokens,
                timeout=request_timeout,
            )
            raw_text = response.choices[0].message.content
            payload = json.loads(raw_text)
            return normalize_teacher_payload(payload), raw_text
        except Exception as error:
            last_error = error
            sleep_seconds = min(10, 2 * (attempt + 1))
            print(
                f"[WARN] Teacher request failed on attempt {attempt + 1}/{max_retries}: {error}. "
                f"Sleeping {sleep_seconds}s."
            )
            time.sleep(sleep_seconds)

    print(f"[ERROR] Teacher request failed after {max_retries} attempts: {last_error}")
    return None, None


def iter_selected_examples(
    input_path: Path,
    start_line: int | None,
    end_line: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    """Load the selected line range from the feature file."""
    selected: list[tuple[int, dict[str, Any]]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if start_line is not None and line_number < start_line:
                continue
            if end_line is not None and line_number > end_line:
                break

            line = line.strip()
            if not line:
                continue
            selected.append((line_number, json.loads(line)))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label judge snapshots with a teacher model."
    )
    parser.add_argument("--input", required=True, help="feature JSONL path.")
    parser.add_argument("--output", required=True, help="Teacher label JSONL path.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Teacher model name.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the teacher API key.",
    )
    parser.add_argument("--start-line", type=int, default=None, help="1-based inclusive start line.")
    parser.add_argument("--end-line", type=int, default=None, help="1-based inclusive end line.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap after line filtering.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Teacher response token budget.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per request.")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.0,
        help="Optional delay between successful requests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file instead of resuming.",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable: {args.api_key_env}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    processed_keys = set() if args.overwrite else load_processed_keys(output_path)
    selected_examples = iter_selected_examples(input_path, args.start_line, args.end_line)

    if args.max_samples is not None:
        selected_examples = selected_examples[: args.max_samples]

    file_mode = "w" if args.overwrite else "a"
    requested = 0
    labeled = 0
    skipped = 0

    with output_path.open(file_mode, encoding="utf-8") as output_file:
        for line_number, example in selected_examples:
            key = (example.get("id"), example.get("turn"))
            if key in processed_keys:
                skipped += 1
                continue

            requested += 1
            payload, raw_text = call_teacher(
                client=client,
                model=args.model,
                user_prompt=build_user_prompt(example),
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                request_timeout=args.request_timeout,
            )
            if payload is None or raw_text is None:
                continue

            output_record = {
                "id": example.get("id"),
                "turn": example.get("turn"),
                "sufficient": payload["sufficient"],
                "gap_items": payload["gap_items"],
                "raw_response": raw_text,
                "teacher_model": args.model,
                "source_line": line_number,
            }
            output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            output_file.flush()
            labeled += 1
            processed_keys.add(key)

            if args.sleep_between_requests > 0:
                time.sleep(args.sleep_between_requests)

    print(f"Selected snapshots: {len(selected_examples)}")
    print(f"Skipped existing labels: {skipped}")
    print(f"Teacher requests sent: {requested}")
    print(f"Successful labels written: {labeled}")
    print(f"Saved labels to: {output_path}")


if __name__ == "__main__":
    main()
