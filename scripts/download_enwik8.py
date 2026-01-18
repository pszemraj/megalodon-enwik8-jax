#!/usr/bin/env python3
"""Download and prepare enwik8.gz into the data directory."""

from __future__ import annotations

import gzip
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://mattmahoney.net/dc/enwik8.zip"


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, dest.open("wb") as f:
        shutil.copyfileobj(response, f)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    gz_path = data_dir / "enwik8.gz"
    if gz_path.exists():
        print(f"{gz_path} already exists.")
        return

    zip_path = data_dir / "enwik8.zip"
    raw_path = data_dir / "enwik8"

    print(f"Downloading {URL} ...")
    _download(URL, zip_path)

    print("Extracting enwik8 ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("enwik8", data_dir)

    print(f"Compressing to {gz_path} ...")
    with raw_path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    zip_path.unlink(missing_ok=True)
    raw_path.unlink(missing_ok=True)

    print("Done.")


if __name__ == "__main__":
    main()
