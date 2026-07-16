import json
import logging
import os
import pathlib
import pickle
import shlex
import subprocess
import sys

try:
    from beir import LoggingHandler
except ImportError:  # pragma: no cover - optional dependency for cleaner logs
    LoggingHandler = logging.StreamHandler

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

OFFICIAL_HOTPOTQA_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip"
)


def load_official_fullwiki_corpus(datasets_dir):
    """Download the official BEIR HotpotQA resources and return the fullwiki paragraph corpus."""
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as error:
        raise ImportError(
            "The `beir` package is required to build the HotpotQA/TriviaQA fullwiki corpus. "
            "Install it before running this script."
        ) from error

    data_path = util.download_and_unzip(OFFICIAL_HOTPOTQA_URL, str(datasets_dir))
    corpus, _, _ = GenericDataLoader(data_path).load(split="test")
    return corpus


def write_jsonl(corpus, jsonl_file):
    """Write a Pyserini JsonCollection from a BEIR-style corpus dict."""
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for doc_id, doc in corpus.items():
            title = (doc.get("title") or "").strip()
            text = (doc.get("text") or doc.get("contents") or "").strip()
            contents = f"{title} {text}".strip()
            f.write(
                json.dumps(
                    {"id": doc_id, "contents": contents},
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_index(jsonl_dir, index_dir, threads=8):
    """Build the Pyserini Lucene index for the provided JsonCollection directory."""
    os.makedirs(index_dir, exist_ok=True)

    cmd = (
        f"{sys.executable} -m pyserini.index "
        f"--collection JsonCollection "
        f"--generator DefaultLuceneDocumentGenerator "
        f"--input {jsonl_dir} "
        f"--index {index_dir} "
        f"--threads {threads} "
        f"--storePositions --storeDocvectors --storeRaw"
    )

    logging.info("Building index with Pyserini:\n%s", cmd)
    subprocess.run(shlex.split(cmd), check=True)


def save_outputs(
    corpus,
    corpus_pkl,
    settings_pkl,
    index_dir,
    k1=0.9,
    b=0.4,
    source="official_fullwiki",
):
    """Persist the retrieval corpus and BM25 settings for downstream inference."""
    corpus_pkl.parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_pkl, "wb") as f:
        pickle.dump(corpus, f)

    with open(settings_pkl, "wb") as f:
        pickle.dump(
            {
                "index_dir": str(index_dir),
                "k1": k1,
                "b": b,
                "source": source,
            },
            f,
        )


def prepare_fullwiki_resources(
    *,
    work_dir,
    resource_tag,
    corpus_filename,
    settings_filename,
    initialize=True,
    threads=8,
    k1=0.9,
    b=0.4,
    source="official_fullwiki",
):
    """
    Build independent full-wiki retrieval resources.

    We reuse the official BEIR HotpotQA Wikipedia corpus because it is an external,
    benchmark-independent full-wiki resource already used in this project for open retrieval.
    """
    work_dir = pathlib.Path(work_dir)
    datasets_dir = work_dir / "datasets"
    jsonl_dir = work_dir / "pyserini_corpus" / resource_tag
    jsonl_file = jsonl_dir / "corpus.jsonl"
    index_dir = work_dir / "pyserini_index" / resource_tag
    corpus_dir = work_dir / "corpus"
    corpus_pkl = corpus_dir / corpus_filename
    settings_pkl = corpus_dir / settings_filename

    logging.info(
        "Preparing independent full-wiki retrieval resources: tag=%s, corpus=%s",
        resource_tag,
        corpus_filename,
    )
    corpus = load_official_fullwiki_corpus(datasets_dir)
    write_jsonl(corpus, jsonl_file)

    if initialize:
        build_index(jsonl_dir, index_dir, threads=threads)
    else:
        logging.info("INITIALIZE=False, skipping index build.")

    save_outputs(
        corpus,
        corpus_pkl,
        settings_pkl,
        index_dir=index_dir,
        k1=k1,
        b=b,
        source=source,
    )
    logging.info("Retrieval resources ready: %s / %s", corpus_pkl, settings_pkl)
