import csv
import json
import os

from datasets import load_dataset

OUTPUT_DIR = "data/original"
TRAIN_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "triviaqa_train.csv")
DEV_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "triviaqa_dev.csv")
TRAIN_SPLIT = "train[0:10000]"
DEV_SPLIT = "validation"


def load_splits():
    """Load the TriviaQA train subset and validation split."""
    train_dataset = load_dataset(
        "mandarjoshi/trivia_qa",
        "rc.wikipedia",
        split=TRAIN_SPLIT,
    )
    dev_dataset = load_dataset(
        "mandarjoshi/trivia_qa",
        "rc.wikipedia",
        split=DEV_SPLIT,
    )
    return train_dataset, dev_dataset


def parse_page_collection(pages, text_key):
    """Return titles and texts from supported page collection formats."""
    titles = []
    texts = []

    if not pages:
        return titles, texts

    if isinstance(pages, dict):
        raw_titles = pages.get("title") or []
        raw_texts = pages.get(text_key) or []
        for title, text in zip(raw_titles, raw_texts):
            clean_title = (title or "").strip()
            if not clean_title:
                continue
            titles.append(clean_title)
            texts.append((text or "").strip())
        return titles, texts

    if isinstance(pages, list):
        for item in pages:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            if not title:
                continue
            text = (item.get(text_key) or "").strip()
            titles.append(title)
            texts.append(text)

    return titles, texts


def extract_answer(entry):
    """Return one normalized answer string."""
    answer_info = entry.get("answer") or {}
    normalized = (answer_info.get("normalized_value") or "").strip()
    if normalized:
        return normalized

    aliases = answer_info.get("aliases") or []
    if aliases:
        return str(aliases[0]).strip()

    return ""


def build_row(entry):
    """Convert one TriviaQA example into a CSV row."""
    qid = entry.get("question_id") or ""
    question = entry.get("question", "")
    answer = extract_answer(entry)

    doc_titles, doc_texts = parse_page_collection(
        entry.get("entity_pages"),
        text_key="wiki_context",
    )

    search_titles, search_texts = parse_page_collection(
        entry.get("search_results"),
        text_key="search_context",
    )

    doc_set = set(doc_titles)
    distractor_docs = []
    distractor_doc_texts = []
    seen_distractors = set()

    for title, text in zip(search_titles, search_texts):
        if title in doc_set or title in seen_distractors:
            continue
        seen_distractors.add(title)
        distractor_docs.append(title)
        distractor_doc_texts.append(text)

    supporting_facts = []

    return [
        qid,
        question,
        answer,
        json.dumps(doc_titles, ensure_ascii=False),
        json.dumps(supporting_facts, ensure_ascii=False),
        json.dumps(doc_texts, ensure_ascii=False),
        json.dumps(distractor_docs, ensure_ascii=False),
        json.dumps(distractor_doc_texts, ensure_ascii=False),
    ]


def save_split_to_csv(data_split, output_file):
    """Write one dataset split to CSV."""
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
    train_dataset, dev_dataset = load_splits()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_split_to_csv(train_dataset, TRAIN_OUTPUT_CSV)
    save_split_to_csv(dev_dataset, DEV_OUTPUT_CSV)

    print(f"Train data saved to {TRAIN_OUTPUT_CSV}")
    print(f"Dev data saved to {DEV_OUTPUT_CSV}")


if __name__ == "__main__":
    main()