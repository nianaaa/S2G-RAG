import argparse
import hashlib
import json
import logging
import pathlib
import subprocess
import sys

from fullwiki_resources import build_index, save_outputs, write_jsonl


INITIALIZE = True
THREADS = 8
K1, B = 0.9, 0.4

WORK_DIR = pathlib.Path(__file__).parent.absolute()
REPO_ROOT = WORK_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "2WikiMultiHopQA" / "data"
DEFAULT_SPLIT_PATHS = (
    DEFAULT_DATA_DIR / "train.json",
    DEFAULT_DATA_DIR / "dev.json",
    DEFAULT_DATA_DIR / "test.json",
)


def iter_context_documents(example):
    """Yield `(title, paragraph_text)` pairs from supported 2Wiki context formats."""
    context = example.get("context")
    if isinstance(context, dict):
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        for title, sent_list in zip(titles, sentences):
            if not title:
                continue
            paragraph_text = " ".join(str(sentence).strip() for sentence in (sent_list or []) if sentence)
            if paragraph_text.strip():
                yield str(title).strip(), paragraph_text.strip()
        return

    if isinstance(context, list):
        for item in context:
            title = None
            sentences = None

            if isinstance(item, (list, tuple)) and len(item) == 2:
                title, sentences = item
            elif isinstance(item, dict):
                title = item.get("title")
                sentences = item.get("sentences") or item.get("text")

            if not title:
                continue

            if isinstance(sentences, list):
                paragraph_text = " ".join(str(sentence).strip() for sentence in sentences if sentence)
            elif isinstance(sentences, str):
                paragraph_text = sentences.strip()
            else:
                paragraph_text = ""

            if paragraph_text:
                yield str(title).strip(), paragraph_text


def make_doc_id(title, paragraph_text):
    """Create a deterministic document id from title and paragraph text."""
    digest = hashlib.sha1(f"{title}\n{paragraph_text}".encode("utf-8")).hexdigest()
    return f"2wiki_{digest}"


def load_2wiki_ircot_corpus(split_paths):
    """
    Build the 2Wiki retrieval corpus in the same spirit as IRCoT.
    """
    corpus = {}
    seen_pairs = set()

    for split_path in split_paths:
        with split_path.open("r", encoding="utf-8") as handle:
            examples = json.load(handle)

        logging.info("Loading 2Wiki contexts from %s (%d examples)", split_path, len(examples))
        for example in examples:
            for title, paragraph_text in iter_context_documents(example):
                key = (title, paragraph_text)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                doc_id = make_doc_id(title, paragraph_text)
                corpus[doc_id] = {
                    "title": title,
                    "text": paragraph_text,
                }

    logging.info("Built 2Wiki corpus with %d unique paragraphs", len(corpus))
    return corpus


def resolve_split_paths(paths, allow_missing=False):
    """Resolve split paths and optionally skip missing files."""
    resolved = []
    missing = []

    for path in paths:
        path = pathlib.Path(path)
        if path.exists():
            resolved.append(path)
        else:
            missing.append(path)

    if missing and not allow_missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing 2Wiki split files required for the retrieval corpus:\n"
            f"{formatted}\n"
            "Provide the files or rerun with --allow-missing-splits."
        )

    for path in missing:
        logging.warning("Skipping missing 2Wiki split file: %s", path)

    if not resolved:
        raise FileNotFoundError("No 2Wiki split files were found.")

    return resolved


def maybe_download_default_2wiki_data(paths, download_missing):
    """Download the official 2Wiki raw files when the default split paths are missing."""
    if not download_missing:
        return

    default_paths = [path.resolve() for path in DEFAULT_SPLIT_PATHS]
    requested_paths = [pathlib.Path(path).resolve() for path in paths]
    if requested_paths != default_paths:
        return

    if all(path.exists() for path in DEFAULT_SPLIT_PATHS):
        return

    downloader = REPO_ROOT / "data" / "download_2wikimultihopqa.py"
    cmd = [
        sys.executable,
        str(downloader),
        "--output-dir",
        str(DEFAULT_DATA_DIR),
    ]
    logging.info("Default 2Wiki split files are missing. Running downloader: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build a 2WikiMultiHopQA BM25 corpus."
    )
    parser.add_argument(
        "--split-paths",
        nargs="*",
        default=[str(path) for path in DEFAULT_SPLIT_PATHS],
        help="2Wiki raw split JSON files to include in the corpus.",
    )
    parser.add_argument(
        "--allow-missing-splits",
        action="store_true",
        help="Skip missing split files instead of failing.",
    )
    parser.add_argument(
        "--download-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically download the official 2Wiki raw files when default split paths are missing.",
    )
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--skip-index-build", action="store_true")
    args = parser.parse_args()

    maybe_download_default_2wiki_data(args.split_paths, download_missing=args.download_missing)
    split_paths = resolve_split_paths(args.split_paths, allow_missing=args.allow_missing_splits)
    corpus = load_2wiki_ircot_corpus(split_paths)

    jsonl_dir = WORK_DIR / "pyserini_corpus" / "2wikimultihopqa_ircot"
    jsonl_file = jsonl_dir / "corpus.jsonl"
    index_dir = WORK_DIR / "pyserini_index" / "2wikimultihopqa_ircot"
    corpus_dir = WORK_DIR / "corpus"
    corpus_pkl = corpus_dir / "2wiki_corpus.pkl"
    settings_pkl = corpus_dir / "2wiki_retriever_settings.pkl"

    write_jsonl(corpus, jsonl_file)
    if not args.skip_index_build:
        build_index(jsonl_dir, index_dir, threads=args.threads)
    else:
        logging.info("Skipping index build because --skip-index-build was set.")

    save_outputs(
        corpus,
        corpus_pkl,
        settings_pkl,
        index_dir=index_dir,
        k1=K1,
        b=B,
        source="ircot_2wikimultihopqa_context_union",
    )
    logging.info("2Wiki retrieval resources ready: %s / %s", corpus_pkl, settings_pkl)


if __name__ == "__main__":
    main()
