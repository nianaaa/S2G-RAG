from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


QUESTION_LINE_PATTERN = re.compile(r"Question:\s*(.*?)(?=\nContext:|\n|$)", re.S)
QUERY_LINE_PATTERN = re.compile(r"Query:\s*(.*?)(?=\n|$)")
RETRIEVED_BLOCK_PATTERN = re.compile(r"Retrieved Document:\s*(.*?)(?=\nQuery:|\Z)", re.S)


def parse_list_field(value: Any) -> list[Any]:
    """Safely parse CSV cells that may store Python-style lists as strings."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def parse_bool(value: Any) -> bool:
    """Normalize common CSV boolean spellings to a Python bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def get_first_present(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-empty value found among candidate column names."""
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def extract_question(task_content: str) -> str:
    """Recover the original question from a trace prompt."""
    if not isinstance(task_content, str):
        return ""

    match = QUESTION_LINE_PATTERN.search(task_content)
    if match:
        question = match.group(1).strip()
        if question:
            return question

    for match in QUERY_LINE_PATTERN.finditer(task_content):
        question = match.group(1).strip()
        if question:
            return question

    return task_content.strip()


def extract_retrieved_context(task_content: str) -> str:
    """Keep only retrieved document blocks for title-coverage heuristics."""
    if not isinstance(task_content, str) or not task_content:
        return ""

    chunks = [match.group(1).strip() for match in RETRIEVED_BLOCK_PATTERN.finditer(task_content)]
    chunks = [chunk for chunk in chunks if chunk]
    return "\n\n".join(chunks)


def compute_doc_coverage(gold_docs: list[str], coverage_context: str) -> float:
    """Approximate title coverage by exact title substring matching."""
    if not gold_docs:
        return 0.0

    hits = 0
    for title in gold_docs:
        if isinstance(title, str) and title and title in coverage_context:
            hits += 1
    return hits / len(gold_docs)


def parse_turn(value: Any) -> int:
    """Convert the turn column to an integer, defaulting to zero on failure."""
    try:
        return int(value)
    except Exception:
        return 0


def build_output_record(
    row: dict[str, Any],
    strong_pos_threshold: float,
    strong_neg_threshold: float,
) -> dict[str, Any] | None:
    """Convert a single trace row into a feature record."""
    task_content = str(
        get_first_present(
            row,
            "Reasoner Task Content",
            "ReasonerTaskContent",
            "Task Content",
            default="",
        )
        or ""
    )
    question = extract_question(task_content)
    coverage_context = extract_retrieved_context(task_content)

    if not coverage_context.strip():
        return None

    gold_docs = parse_list_field(
        get_first_present(row, "Gold Retrieved Docs", "Gold Retrieved Doc", default=None)
    )
    gold_answers = parse_list_field(get_first_present(row, "Gold Answers", "Gold Answer", default=None))
    retrieved_titles = parse_list_field(
        get_first_present(row, "Retrieved Titles", "Retrieved Title", default=None)
    )
    retrieved_docs = parse_list_field(
        get_first_present(row, "Retrieved Docs", "Retrieved Doc", default=None)
    )

    gold_answer = gold_answers[0] if gold_answers else ""
    doc_coverage = compute_doc_coverage(gold_docs, coverage_context)
    answer_in_context = bool(gold_answer) and str(gold_answer) in coverage_context
    missing_gold_docs = [
        title
        for title in gold_docs
        if isinstance(title, str) and title and title not in coverage_context
    ]

    teacher_verdict_raw = get_first_present(
        row,
        "Verdict",
        "Teacher Verdict",
        "Correct Answer",
        default=0,
    )
    try:
        teacher_verdict = int(teacher_verdict_raw)
    except Exception:
        teacher_verdict = 0

    return {
        "id": get_first_present(row, "ID", "Id", "id", "qid", "QID", default=None),
        "turn": parse_turn(get_first_present(row, "Turn", "turn", "Round", "round", default=0)),
        "question": question,
        "context": task_content,
        "coverage_context": coverage_context,
        "gold_answer": gold_answer,
        "gold_docs": gold_docs,
        "retrieved_titles": retrieved_titles,
        "retrieved_docs": retrieved_docs,
        "teacher_verdict": teacher_verdict,
        "correct_retrieval": parse_bool(
            get_first_present(row, "Correct Retrieval", "correct_retrieval", default=False)
        ),
        "doc_coverage": doc_coverage,
        "missing_gold_docs": missing_gold_docs,
        "answer_in_context": answer_in_context,
        "rule_strong_pos_candidate": doc_coverage >= strong_pos_threshold and answer_in_context,
        "rule_strong_neg_candidate": doc_coverage <= strong_neg_threshold and not answer_in_context,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build judge training features from an inference trace CSV."
    )
    parser.add_argument("--input", required=True, help="Path to the inference trace CSV.")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file.")
    parser.add_argument(
        "--strong-pos-threshold",
        type=float,
        default=0.7,
        help="Coverage threshold for strong positive rule candidates.",
    )
    parser.add_argument(
        "--strong-neg-threshold",
        type=float,
        default=0.1,
        help="Coverage threshold for strong negative rule candidates.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    written_rows = 0
    skipped_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as input_file, output_path.open(
        "w", encoding="utf-8"
    ) as output_file:
        reader = csv.DictReader(input_file)

        for row in reader:
            total_rows += 1
            output_record = build_output_record(
                row=row,
                strong_pos_threshold=args.strong_pos_threshold,
                strong_neg_threshold=args.strong_neg_threshold,
            )
            if output_record is None:
                skipped_rows += 1
                continue

            output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            written_rows += 1

    print(f"Loaded rows: {total_rows}")
    print(f"Written feature rows: {written_rows}")
    print(f"Skipped rows without retrieved context: {skipped_rows}")
    print(f"Saved features to: {output_path}")


if __name__ == "__main__":
    main()
