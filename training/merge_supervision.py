from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path, source_name: str) -> list[dict]:
    """Load a JSONL supervision file and attach a source field when missing."""
    examples: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record.setdefault("source", source_name)
            examples.append(record)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple cleaned supervision files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input cleaned supervision JSONL files.",
    )
    parser.add_argument(
        "--source-names",
        nargs="*",
        default=None,
        help="Optional source names aligned with --inputs. Defaults to file stems.",
    )
    parser.add_argument("--output", required=True, help="Merged output JSONL path.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle merged records before writing.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.inputs]
    if args.source_names is not None and len(args.source_names) not in {0, len(input_paths)}:
        raise ValueError("--source-names must be omitted or have the same length as --inputs.")

    if args.source_names:
        source_names = args.source_names
    else:
        source_names = [path.stem for path in input_paths]

    merged: list[dict] = []
    for path, source_name in zip(input_paths, source_names):
        examples = load_jsonl(path, source_name)
        merged.extend(examples)
        print(f"Loaded {len(examples)} examples from {path} (source={source_name})")

    if args.shuffle:
        random.Random(args.seed).shuffle(merged)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Total merged examples: {len(merged)}")
    print(f"Saved merged supervision to: {output_path}")


if __name__ == "__main__":
    main()
