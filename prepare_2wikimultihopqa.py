import csv
import json
import os
import pathlib
import subprocess
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "2WikiMultiHopQA" / "data"
TRAIN_INPUT_JSON = str(DATA_DIR / "train.json")
DEV_INPUT_JSON = str(DATA_DIR / "dev.json")
OUTPUT_DIR = "data/original"
TRAIN_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "2wikimultihopqa_train.csv")
DEV_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "2wikimultihopqa_dev.csv")
# TRAIN_LIMIT = 10000


def ensure_raw_2wiki_data():
    """Download the official 2Wiki raw files if they are not available locally."""
    required_files = [TRAIN_INPUT_JSON, DEV_INPUT_JSON]
    if all(os.path.exists(path) for path in required_files):
        return

    downloader = SCRIPT_DIR / "download_2wikimultihopqa.py"
    subprocess.run(
        [sys.executable, str(downloader), "--output-dir", str(DATA_DIR)],
        check=True,
    )


def load_json(path, limit=None):
    """Load a JSON list from disk and optionally keep only the first N entries."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is not None:
        return data[:limit]
    return data


def parse_context(context):
    """Return a title-to-sentences mapping from supported 2Wiki context formats."""
    title_to_sentences = {}
    ordered_titles = []

    if isinstance(context, dict):
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        for title, sent_list in zip(titles, sentences):
            if not title:
                continue
            ordered_titles.append(title)
            title_to_sentences[title] = sent_list or []
        return ordered_titles, title_to_sentences

    if isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                title, sent_list = item
            elif isinstance(item, dict):
                title = item.get("title")
                sent_list = item.get("sentences") or item.get("text") or []
            else:
                continue

            if not title:
                continue

            ordered_titles.append(title)
            title_to_sentences[title] = sent_list or []

    return ordered_titles, title_to_sentences


def parse_supporting_facts(supporting_facts):
    """Return a normalized list of [title, sent_id] pairs."""
    pairs = []

    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title") or []
        sent_ids = supporting_facts.get("sent_id") or []
        for title, sent_id in zip(titles, sent_ids):
            if title is None or sent_id is None:
                continue
            pairs.append([title, int(sent_id)])
        return pairs

    if isinstance(supporting_facts, list):
        for item in supporting_facts:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, sent_id = item[0], item[1]
            elif isinstance(item, dict):
                title = item.get("title")
                sent_id = item.get("sent_id")
            else:
                continue

            if title is None or sent_id is None:
                continue
            pairs.append([title, int(sent_id)])

    return pairs


def build_row(entry):
    """Convert one 2Wiki example into a CSV row."""
    qid = entry.get("_id") or entry.get("id") or ""
    question = entry.get("question", "")
    answer = entry.get("answer", "")

    ordered_titles, title_to_sentences = parse_context(entry.get("context"))
    supporting_facts = parse_supporting_facts(entry.get("supporting_facts"))

    gold_docs = list(dict.fromkeys(title for title, _ in supporting_facts))
    gold_set = set(gold_docs)
    distractor_docs = [title for title in ordered_titles if title not in gold_set]

    supporting_texts = []
    for title, sent_id in supporting_facts:
        sentences = title_to_sentences.get(title)
        if sentences is not None and 0 <= sent_id < len(sentences):
            supporting_texts.append(sentences[sent_id])
        else:
            supporting_texts.append(None)

    distractor_doc_texts = [
        title_to_sentences.get(title, [])
        for title in distractor_docs
    ]

    return [
        qid,
        question,
        answer,
        json.dumps(gold_docs, ensure_ascii=False),
        json.dumps(supporting_facts, ensure_ascii=False),
        json.dumps(supporting_texts, ensure_ascii=False),
        json.dumps(distractor_docs, ensure_ascii=False),
        json.dumps(distractor_doc_texts, ensure_ascii=False),
    ]


def save_split_to_csv(data_split, output_file):
    """Write one split to CSV."""
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "ID",
                "Question",
                "Answer",
                "Documents",
                "SupportingFacts",
                "SupportingTexts",
                "DistractorDocs",
                "DistractorDocTexts",
            ]
        )

        for entry in data_split:
            writer.writerow(build_row(entry))


def main():
    ensure_raw_2wiki_data()
    # train_dataset = load_json(TRAIN_INPUT_JSON, limit=TRAIN_LIMIT)
    train_dataset = load_json(TRAIN_INPUT_JSON)
    dev_dataset = load_json(DEV_INPUT_JSON)

    print(len(train_dataset))
    print(len(dev_dataset))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_split_to_csv(train_dataset, TRAIN_OUTPUT_CSV)
    save_split_to_csv(dev_dataset, DEV_OUTPUT_CSV)

    print(f"Train data saved to {TRAIN_OUTPUT_CSV}")
    print(f"Dev data saved to {DEV_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
