from __future__ import annotations

import argparse
import pathlib
import shutil
import urllib.request
import zipfile


OFFICIAL_DATA_URL = (
    "https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/"
    "data_ids_april7.zip?rlkey=u868q6h0jojw4djjg7ea65j46&dl=1"
)
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "2WikiMultiHopQA" / "data"
REQUIRED_FILES = ("train.json", "dev.json", "test.json")
OPTIONAL_FILES = ("id_aliases.json",)


def required_paths(output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the paths that must exist after a successful download."""
    return [output_dir / name for name in REQUIRED_FILES]


def dataset_is_ready(output_dir: pathlib.Path) -> bool:
    """Check whether all required split files already exist."""
    return all(path.exists() for path in required_paths(output_dir))


def download_file(url: str, destination: pathlib.Path) -> None:
    """Download a file using the Python standard library."""
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_archive(archive_path: pathlib.Path, output_dir: pathlib.Path, overwrite: bool) -> None:
    """Extract the official 2Wiki archive into the target directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_name = pathlib.PurePosixPath(member.filename).name
            if member_name not in REQUIRED_FILES + OPTIONAL_FILES:
                continue

            target_path = output_dir / member_name
            if target_path.exists() and not overwrite:
                continue

            with archive.open(member) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def ensure_2wiki_dataset(
    output_dir: pathlib.Path,
    url: str = OFFICIAL_DATA_URL,
    overwrite: bool = False,
    keep_archive: bool = False,
) -> list[pathlib.Path]:
    """Download and extract the official 2Wiki data when files are missing."""
    output_dir = pathlib.Path(output_dir)
    if dataset_is_ready(output_dir) and not overwrite:
        return required_paths(output_dir)

    download_dir = output_dir.parent / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / "2wikimultihopqa_data_ids_april7.zip"

    if overwrite or not archive_path.exists():
        print(f"Downloading official 2WikiMultiHopQA data to {archive_path}")
        download_file(url, archive_path)
    else:
        print(f"Reusing existing archive: {archive_path}")

    print(f"Extracting 2WikiMultiHopQA files into {output_dir}")
    extract_archive(archive_path, output_dir, overwrite=overwrite)

    missing = [path for path in required_paths(output_dir) if not path.exists()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "The 2Wiki download completed, but some required files are still missing:\n"
            f"{formatted}"
        )

    if not keep_archive and archive_path.exists():
        archive_path.unlink()
        if not any(download_dir.iterdir()):
            download_dir.rmdir()

    return required_paths(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the official 2WikiMultiHopQA raw json files."
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where train/dev/test.json will be placed.",
    )
    parser.add_argument(
        "--url",
        default=OFFICIAL_DATA_URL,
        help="Override the official dataset archive URL.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload and overwrite any existing split files.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zip file after extraction.",
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    files = ensure_2wiki_dataset(
        output_dir=output_dir,
        url=args.url,
        overwrite=args.overwrite,
        keep_archive=args.keep_archive,
    )

    print("2WikiMultiHopQA dataset is ready:")
    for path in files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
