from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def normalize_category(value: Any) -> str:
    """Map noisy category spellings to a small controlled set."""
    if not isinstance(value, str):
        return "other"

    normalized = value.strip().lower().replace(" ", "_")
    mapping = {
        "bridge": "bridge_entity",
        "bridge_entity": "bridge_entity",
        "bridge_entity?": "bridge_entity",
        "alias": "bridge_entity",
        "attribute": "attribute",
        "relation": "relation",
        "evidence": "evidence_span",
        "evidence_span": "evidence_span",
        "evidence_span?": "evidence_span",
        "evidence_span_missing": "evidence_span",
        "evidence_span_missing?": "evidence_span",
        "other": "other",
    }
    return mapping.get(normalized, "other")


def normalize_slot(value: Any) -> str:
    """Canonicalize a few recurring slot aliases without over-normalizing."""
    if not isinstance(value, str):
        return "other"

    original = value.strip()
    normalized = original.lower().replace(" ", "_")
    mapping = {
        "foundedyear": "founded_year",
        "founded_year": "founded_year",
        "year_founded": "founded_year",
        "birthdate": "birth_date",
        "birth_date": "birth_date",
        "birth_year": "birth_date",
        "grand_slam_titles": "grand_slam_titles_count",
        "grand_slam_titles_count": "grand_slam_titles_count",
        "grand_slam_titles_won": "grand_slam_titles_count",
        "grand_slam_titles_total": "grand_slam_titles_count",
    }
    return mapping.get(normalized, original)


def extract_gap_items(record: dict[str, Any]) -> list[Any]:
    """Read both the new and older gap-field spellings."""
    for key in ("gap_items", "gap items", "missing_facts", "missing_info"):
        value = record.get(key)
        if isinstance(value, list):
            return value
    return []


def normalize_gap_items(record: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize gap items and drop malformed entries."""
    normalized_items: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for item in extract_gap_items(record):
        if not isinstance(item, dict):
            continue

        category = normalize_category(item.get("category"))
        slot = normalize_slot(item.get("slot"))
        target = str(item.get("target", "") or "").strip() or "Answer"
        description = str(item.get("description", "") or "").strip()

        key = (category, target, slot, description)
        if key in seen:
            continue
        seen.add(key)

        normalized_items.append(
            {
                "category": category,
                "target": target,
                "slot": slot,
                "description": description,
            }
        )

    return normalized_items


def normalize_turn(value: Any) -> int:
    """Make sure feature and label keys use the same turn type."""
    try:
        return int(value)
    except Exception:
        return 0


def load_feature_map(path: Path) -> dict[tuple[Any, int], dict[str, Any]]:
    """Load feature records keyed by `(id, turn)`."""
    feature_map: dict[tuple[Any, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            feature_map[(record.get("id"), normalize_turn(record.get("turn")))] = record
    return feature_map


def split_pos_neg(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split samples by the final sufficiency label."""
    positives = [sample for sample in samples if sample.get("sufficient", False)]
    negatives = [sample for sample in samples if not sample.get("sufficient", False)]
    return positives, negatives


def balance_basic(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    ratio: float,
    mode: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Balance positives and negatives by downsampling or upsampling."""
    rng = random.Random(seed)
    stats = {
        "before_pos": len(positives),
        "before_neg": len(negatives),
        "after_pos": len(positives),
        "after_neg": len(negatives),
        "mode": mode,
        "target_ratio": ratio,
    }

    if mode == "none" or ratio <= 0 or not positives or not negatives:
        return positives, negatives, stats

    current_ratio = len(positives) / len(negatives)

    if current_ratio > ratio:
        target_pos = max(1, min(len(positives), int(round(ratio * len(negatives)))))
        positives = rng.sample(positives, target_pos)
    elif current_ratio < ratio:
        target_neg = max(1, min(len(negatives), int(round(len(positives) / ratio))))
        negatives = rng.sample(negatives, target_neg)

    if mode == "upsample":
        current_ratio = len(positives) / len(negatives)
        if current_ratio < ratio and positives:
            target_pos = int(round(ratio * len(negatives)))
            positives = positives + [rng.choice(positives) for _ in range(max(0, target_pos - len(positives)))]
        elif current_ratio > ratio and negatives:
            target_neg = int(round(len(positives) / ratio))
            negatives = negatives + [rng.choice(negatives) for _ in range(max(0, target_neg - len(negatives)))]

    stats["after_pos"] = len(positives)
    stats["after_neg"] = len(negatives)
    return positives, negatives, stats


def balance_samples(
    samples: list[dict[str, Any]],
    ratio: float,
    mode: str,
    seed: int,
    scope: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Balance either all samples or only the hard subset."""
    rng = random.Random(seed)

    if scope == "hard_only":
        hard = [sample for sample in samples if sample.get("sample_type") == "hard"]
        protected = [sample for sample in samples if sample.get("sample_type") != "hard"]
        positives, negatives = split_pos_neg(hard)
        positives, negatives, stats = balance_basic(positives, negatives, ratio, mode, seed)
        merged = positives + negatives + protected
        rng.shuffle(merged)
        stats["scope"] = scope
        stats["protected_examples"] = len(protected)
        return merged, stats

    positives, negatives = split_pos_neg(samples)
    positives, negatives, stats = balance_basic(positives, negatives, ratio, mode, seed)
    merged = positives + negatives
    rng.shuffle(merged)
    stats["scope"] = scope
    return merged, stats


def balance_by_turn(
    samples: list[dict[str, Any]],
    ratio: float,
    mode: str,
    seed: int,
    scope: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply balancing inside each turn bucket before shuffling the result."""
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        buckets[normalize_turn(sample.get("turn"))].append(sample)

    merged: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    for turn, bucket in sorted(buckets.items()):
        balanced_bucket, bucket_stats = balance_samples(
            samples=bucket,
            ratio=ratio,
            mode=mode,
            seed=seed + turn,
            scope=scope,
        )
        merged.extend(balanced_bucket)
        stats[f"turn_{turn}"] = bucket_stats

    random.Random(seed).shuffle(merged)
    return merged, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge and clean teacher supervision."
    )
    parser.add_argument("--features", required=True, help="feature JSONL path.")
    parser.add_argument("--labels", required=True, help="Teacher label JSONL path.")
    parser.add_argument("--output", required=True, help="Cleaned supervision JSONL path.")
    parser.add_argument(
        "--drop-empty-context",
        action="store_true",
        help="Drop examples whose context becomes empty after loading.",
    )
    parser.add_argument(
        "--keep-empty-gap-items",
        action="store_true",
        help="Keep insufficient examples even if the teacher returned no gap items.",
    )
    parser.add_argument(
        "--pos-neg-ratio",
        type=float,
        default=1.0,
        help="Target positive/negative ratio used when balancing is enabled.",
    )
    parser.add_argument(
        "--balance-mode",
        choices=["none", "downsample", "upsample"],
        default="none",
        help="Whether to rebalance the cleaned supervision set.",
    )
    parser.add_argument(
        "--balance-scope",
        choices=["all", "hard_only"],
        default="hard_only",
        help="Apply balancing to all examples or only the hard subset.",
    )
    parser.add_argument(
        "--balance-by-turn",
        action="store_true",
        help="Balance each turn bucket independently.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    feature_path = Path(args.features)
    label_path = Path(args.labels)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_map = load_feature_map(feature_path)
    stats = Counter()
    cleaned_samples: list[dict[str, Any]] = []

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            label = json.loads(line)
            key = (label.get("id"), normalize_turn(label.get("turn")))
            feature = feature_map.get(key)

            if feature is None:
                stats["dropped_missing_feature"] += 1
                continue

            sufficient = label.get("sufficient")
            if not isinstance(sufficient, bool):
                stats["dropped_invalid_sufficient"] += 1
                continue

            context = str(feature.get("context", "") or "")
            if args.drop_empty_context and not context.split():
                stats["dropped_empty_context"] += 1
                continue

            strong_pos = bool(feature.get("rule_strong_pos_candidate", False))
            strong_neg = bool(feature.get("rule_strong_neg_candidate", False))

            if strong_pos and sufficient is False:
                stats["dropped_conflict_strong_pos"] += 1
                continue
            if strong_neg and sufficient is True:
                stats["dropped_conflict_strong_neg"] += 1
                continue

            gap_items = [] if sufficient else normalize_gap_items(label)
            if not sufficient and not gap_items and not args.keep_empty_gap_items:
                stats["dropped_empty_gap_items"] += 1
                continue

            if sufficient and strong_pos:
                sample_type = "strong_pos"
            elif (not sufficient) and strong_neg:
                sample_type = "strong_neg"
            else:
                sample_type = "hard"

            output_record = {
                "id": feature.get("id"),
                "turn": normalize_turn(feature.get("turn")),
                "question": feature.get("question", ""),
                "context": context,
                "gold_answer": feature.get("gold_answer", ""),
                "gold_docs": feature.get("gold_docs", []),
                "doc_coverage": feature.get("doc_coverage", 0.0),
                "missing_gold_docs": feature.get("missing_gold_docs", []),
                "answer_in_context": feature.get("answer_in_context", False),
                "teacher_verdict": feature.get("teacher_verdict", 0),
                "correct_retrieval": feature.get("correct_retrieval", False),
                "rule_strong_pos_candidate": strong_pos,
                "rule_strong_neg_candidate": strong_neg,
                "sufficient": sufficient,
                "gap_items": gap_items,
                "sample_type": sample_type,
            }

            cleaned_samples.append(output_record)
            stats["kept_raw"] += 1
            stats["kept_raw_pos" if sufficient else "kept_raw_neg"] += 1
            stats[f"kept_raw_turn_{output_record['turn']}"] += 1

    if args.balance_mode == "none":
        final_samples = cleaned_samples
        balance_stats: dict[str, Any] = {"mode": "none"}
    elif args.balance_by_turn:
        final_samples, balance_stats = balance_by_turn(
            samples=cleaned_samples,
            ratio=args.pos_neg_ratio,
            mode=args.balance_mode,
            seed=args.seed,
            scope=args.balance_scope,
        )
    else:
        final_samples, balance_stats = balance_samples(
            samples=cleaned_samples,
            ratio=args.pos_neg_ratio,
            mode=args.balance_mode,
            seed=args.seed,
            scope=args.balance_scope,
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for sample in final_samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("Finished cleaning supervision.")
    print(f"Loaded features: {len(feature_map)}")
    print(f"Kept samples before balancing: {stats.get('kept_raw', 0)}")
    print(f"Final samples written: {len(final_samples)}")
    print("Filter stats:")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    print("Balance stats:")
    print(json.dumps(balance_stats, ensure_ascii=False, indent=2))
    print(f"Saved cleaned supervision to: {output_path}")


if __name__ == "__main__":
    main()
