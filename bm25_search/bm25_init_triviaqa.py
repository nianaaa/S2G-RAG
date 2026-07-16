import argparse
import hashlib
import logging
import pathlib
import re

from fullwiki_resources import build_index, save_outputs, write_jsonl


THREADS = 8
K1, B = 0.9, 0.4
DEFAULT_CHUNK_MAX_CHARS = 800
DEFAULT_CHUNK_OVERLAP_SENTENCES = 0

WORK_DIR = pathlib.Path(__file__).parent.absolute()
DEFAULT_CONFIG_NAME = "rc.wikipedia"
PREFERRED_SPLITS = ("train", "validation", "test")


def parse_page_collection(pages, text_key):
    """Return `(title, text)` pairs from supported TriviaQA page formats."""
    if not pages:
        return

    if isinstance(pages, dict):
        titles = pages.get("title") or []
        texts = pages.get(text_key) or []
        for title, text in zip(titles, texts):
            clean_title = str(title or "").strip()
            clean_text = str(text or "").strip()
            if clean_title and clean_text:
                yield clean_title, clean_text
        return

    if isinstance(pages, list):
        for item in pages:
            if not isinstance(item, dict):
                continue
            clean_title = str(item.get("title") or "").strip()
            clean_text = str(item.get(text_key) or "").strip()
            if clean_title and clean_text:
                yield clean_title, clean_text


def iter_triviaqa_documents(example):
    """
    Yield retrieval passages from one TriviaQA example.

    To mirror the 2Wiki setup, we build a benchmark-derived corpus from the
    passages already attached to TriviaQA examples:
    - `entity_pages` as supporting pages
    - `search_results` as distractor pages
    """
    seen = set()

    for title, text in parse_page_collection(example.get("entity_pages"), text_key="wiki_context"):
        key = (title, text)
        if key in seen:
            continue
        seen.add(key)
        yield title, text

    for title, text in parse_page_collection(example.get("search_results"), text_key="search_context"):
        key = (title, text)
        if key in seen:
            continue
        seen.add(key)
        yield title, text


def make_doc_id(title, text):
    """Create a deterministic document id from title and paragraph text."""
    digest = hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()
    return f"triviaqa_{digest}"


def _normalize_whitespace(text):
    """Collapse repeated whitespace while preserving readable sentence boundaries."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _chunk_long_text(text, max_chars):
    """Fallback character-based chunking for very long text spans without clear sentence boundaries."""
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    step = max(1, max_chars - max(80, max_chars // 5))
    chunks = []
    start = 0
    while start < len(cleaned):
        chunk = cleaned[start : start + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        if start + max_chars >= len(cleaned):
            break
        start += step
    return chunks


def split_text_into_chunks(text, max_chars=DEFAULT_CHUNK_MAX_CHARS, overlap_sentences=DEFAULT_CHUNK_OVERLAP_SENTENCES):
    """Split TriviaQA page text into shorter sentence-aware chunks for retrieval."""
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part and part.strip()]
    if len(sentences) <= 1:
        return _chunk_long_text(cleaned, max_chars=max_chars)

    chunks = []
    start = 0
    while start < len(sentences):
        current = []
        current_len = 0
        end = start

        while end < len(sentences):
            sentence = sentences[end]
            sentence_len = len(sentence) + (1 if current else 0)

            if current and current_len + sentence_len > max_chars:
                break

            if not current and len(sentence) > max_chars:
                chunks.extend(_chunk_long_text(sentence, max_chars=max_chars))
                end += 1
                break

            current.append(sentence)
            current_len += sentence_len
            end += 1

        if current:
            chunks.append(" ".join(current).strip())

        if end >= len(sentences):
            break

        next_start = max(start + 1, end - max(0, overlap_sentences))
        if next_start <= start:
            next_start = end
        start = next_start

    deduped = []
    seen = set()
    for chunk in chunks:
        if chunk and chunk not in seen:
            deduped.append(chunk)
            seen.add(chunk)
    return deduped


def load_dataset_modules():
    """Import HuggingFace datasets lazily so `--help` works without the dependency."""
    try:
        from datasets import load_dataset, load_dataset_builder
    except ImportError as error:
        raise ImportError(
            "The `datasets` package is required to build the TriviaQA retrieval corpus. "
            "Install it before running this script."
        ) from error
    return load_dataset, load_dataset_builder


def resolve_available_splits(config_name, requested_splits, allow_missing):
    """Keep only dataset splits that exist for the selected TriviaQA config."""
    _, load_dataset_builder = load_dataset_modules()
    builder = load_dataset_builder("mandarjoshi/trivia_qa", config_name)
    available = set(builder.info.splits.keys())

    selected = []
    missing = []
    for split in requested_splits:
        if split in available:
            selected.append(split)
        else:
            missing.append(split)

    if missing and not allow_missing:
        raise ValueError(
            f"Requested TriviaQA splits are unavailable for config `{config_name}`: {missing}. "
            f"Available splits: {sorted(available)}"
        )

    for split in missing:
        logging.warning("Skipping unavailable TriviaQA split: %s", split)

    if not selected:
        raise ValueError(
            f"No valid TriviaQA splits selected for config `{config_name}`. "
            f"Available splits: {sorted(available)}"
        )

    return selected


def load_triviaqa_union_corpus(config_name, splits, chunk_max_chars, chunk_overlap_sentences):
    """Build the TriviaQA corpus from the union of chunked passages attached to all selected splits."""
    load_dataset, _ = load_dataset_modules()

    corpus = {}
    seen_pairs = set()
    source_pages = 0

    for split in splits:
        dataset = load_dataset("mandarjoshi/trivia_qa", config_name, split=split)
        logging.info("Loading TriviaQA split %s (%d examples)", split, len(dataset))

        for example in dataset:
            for title, text in iter_triviaqa_documents(example):
                key = (title, text)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                source_pages += 1

                chunks = split_text_into_chunks(
                    text,
                    max_chars=chunk_max_chars,
                    overlap_sentences=chunk_overlap_sentences,
                )
                for chunk_text in chunks:
                    doc_id = make_doc_id(title, chunk_text)
                    corpus[doc_id] = {
                        "title": title,
                        "text": chunk_text,
                    }

    logging.info(
        "Built TriviaQA chunked corpus with %d source pages and %d unique chunks",
        source_pages,
        len(corpus),
    )
    return corpus


def main():
    parser = argparse.ArgumentParser(
        description="Build a TriviaQA BM25 corpus from chunked dataset-provided passages."
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="TriviaQA dataset config name, e.g. `rc.wikipedia`.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(PREFERRED_SPLITS),
        help="Dataset splits to union into the retrieval corpus.",
    )
    parser.add_argument(
        "--allow-missing-splits",
        action="store_true",
        help="Skip unavailable splits instead of failing.",
    )
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=DEFAULT_CHUNK_MAX_CHARS,
        help="Maximum characters per TriviaQA retrieval chunk.",
    )
    parser.add_argument(
        "--chunk-overlap-sentences",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_SENTENCES,
        help="How many trailing sentences to overlap between adjacent chunks.",
    )
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--skip-index-build", action="store_true")
    args = parser.parse_args()

    splits = resolve_available_splits(
        config_name=args.config_name,
        requested_splits=args.splits,
        allow_missing=args.allow_missing_splits,
    )
    corpus = load_triviaqa_union_corpus(
        config_name=args.config_name,
        splits=splits,
        chunk_max_chars=args.chunk_max_chars,
        chunk_overlap_sentences=args.chunk_overlap_sentences,
    )

    jsonl_dir = WORK_DIR / "pyserini_corpus" / "triviaqa_union"
    jsonl_file = jsonl_dir / "corpus.jsonl"
    index_dir = WORK_DIR / "pyserini_index" / "triviaqa_union"
    corpus_dir = WORK_DIR / "corpus"
    corpus_pkl = corpus_dir / "triviaqa_corpus.pkl"
    settings_pkl = corpus_dir / "triviaqa_retriever_settings.pkl"

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
        source="triviaqa_entity_pages_search_results_chunked_union",
    )
    logging.info("TriviaQA retrieval resources ready: %s / %s", corpus_pkl, settings_pkl)


if __name__ == "__main__":
    main()
