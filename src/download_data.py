"""Download the UCI Heart Disease dataset.

Downloads the four 'processed' data files (Cleveland, Hungarian, Switzerland,
VA) from the UCI Machine Learning Repository into ``data/raw/``.

If the download fails (e.g. no internet), it falls back to the local copies
bundled in the ``heart+disease/`` folder so the project still works offline.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import requests

# Project paths (this file lives in src/, so the project root is its parent's parent)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOCAL_SOURCE_DIR = PROJECT_ROOT / "heart+disease"

# UCI repository base URL for the heart-disease database
UCI_BASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
)

# The four hospital data files we need
DATA_FILES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
]


def download_file(filename: str, dest_dir: Path) -> bool:
    """Try to download one file from UCI. Return True on success."""
    url = f"{UCI_BASE_URL}/{filename}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        (dest_dir / filename).write_bytes(response.content)
        print(f"  downloaded: {filename}")
        return True
    except Exception as exc:  # noqa: BLE001 - we want any failure to trigger fallback
        print(f"  download failed for {filename}: {exc}")
        return False


def copy_local_file(filename: str, dest_dir: Path) -> bool:
    """Fall back to the local bundled copy. Return True on success."""
    local_path = LOCAL_SOURCE_DIR / filename
    if local_path.exists():
        shutil.copy(local_path, dest_dir / filename)
        print(f"  copied local copy: {filename}")
        return True
    print(f"  no local copy found for {filename}")
    return False


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving raw data to: {RAW_DIR}")

    for filename in DATA_FILES:
        if not download_file(filename, RAW_DIR):
            copy_local_file(filename, RAW_DIR)

    print("Done. Raw files in data/raw/:")
    for path in sorted(RAW_DIR.glob("*.data")):
        print(f"  {path.name} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
