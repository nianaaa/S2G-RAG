import csv
import json
import os

from datasets import load_dataset

OUTPUT_DIR = "data/original"
TRAIN_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "hotpotqa_train.csv")
DEV_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "hotpotqa_dev.csv")


def load_splits():
    """Load the HotpotQA train subset and validation split."""
    train_dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="train[0:10000]",
    )
    dev_dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
    )
    return train_dataset, dev_dataset


def build_row(entry):
    """Convert one HotpotQA example into a CSV row."""
    qid = entry["id"]
    question = entry.get("question", "")
    answer = entry.get("answer", "")

    context = entry["context"]
    ctx_titles = context["title"]
    ctx_sentences = context["sentences"]
    title_to_sentences = {
        title: sentences
        for title, sentences in zip(ctx_titles, ctx_sentences)
    }

    supporting_facts = entry["supporting_facts"]
    sf_titles = supporting_facts["title"]
    sf_sent_ids = supporting_facts["sent_id"]

    gold_docs = list(dict.fromkeys(sf_titles))
    gold_set = set(gold_docs)
    distractor_docs = [title for title in ctx_titles if title not in gold_set]

    supporting = []
    supporting_texts = []
    for title, sent_id in zip(sf_titles, sf_sent_ids):
        sent_id = int(sent_id)
        supporting.append([title, sent_id])

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
        json.dumps(supporting, ensure_ascii=False),
        json.dumps(supporting_texts, ensure_ascii=False),
        json.dumps(distractor_docs, ensure_ascii=False),
        json.dumps(distractor_doc_texts, ensure_ascii=False),
    ]


def save_split_to_csv(data_split, output_file):
    """Write one dataset split to CSV."""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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

    print(len(train_dataset))
    print(len(dev_dataset))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_split_to_csv(train_dataset, TRAIN_OUTPUT_CSV)
    save_split_to_csv(dev_dataset, DEV_OUTPUT_CSV)

    print(f"Train data saved to {TRAIN_OUTPUT_CSV}")
    print(f"Dev data saved to {DEV_OUTPUT_CSV}")


if __name__ == "__main__":
    main()