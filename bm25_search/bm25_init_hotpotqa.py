import argparse
import pathlib

from fullwiki_resources import prepare_fullwiki_resources


THREADS = 8
K1, B = 0.9, 0.4

WORK_DIR = pathlib.Path(__file__).parent.absolute()


def main():
    parser = argparse.ArgumentParser(
        description="Build the HotpotQA BM25 index from the official HotpotQA fullwiki corpus."
    )
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--skip-index-build", action="store_true")
    args = parser.parse_args()

    prepare_fullwiki_resources(
        work_dir=WORK_DIR,
        resource_tag="hotpotqa_ircot_fullwiki",
        corpus_filename="corpus.pkl",
        settings_filename="retriever_settings.pkl",
        initialize=not args.skip_index_build,
        threads=args.threads,
        k1=K1,
        b=B
    )


if __name__ == "__main__":
    main()
