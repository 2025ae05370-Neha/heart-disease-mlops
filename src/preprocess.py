"""Clean and merge the raw UCI Heart Disease files into a single CSV.

Steps performed:
1. Read the four raw hospital files from ``data/raw/``.
2. Attach proper column names (the raw files have no header).
3. Tag each row with its ``source`` hospital.
4. Replace the ``?`` placeholder with ``NaN`` and coerce all features to numbers.
5. Binarize the target: ``num`` 0 -> 0 (no disease), 1-4 -> 1 (disease).
6. Save the merged, cleaned table to ``data/processed/heart.csv``.

Missing values are intentionally left as ``NaN`` here so the EDA notebook can
visualize them. Imputation is handled later inside the model pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "heart.csv"

# The 13 features + original target column, in file order.
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",  # original 0-4 target (becomes binary 'target' below)
]

# Map each raw file to a short hospital name.
SOURCE_FILES = {
    "processed.cleveland.data": "cleveland",
    "processed.hungarian.data": "hungarian",
    "processed.switzerland.data": "switzerland",
    "processed.va.data": "va",
}


def load_raw() -> pd.DataFrame:
    """Read all raw files and concatenate them into one DataFrame."""
    frames = []
    for filename, source in SOURCE_FILES.items():
        path = RAW_DIR / filename
        if not path.exists():
            print(f"  WARNING: missing {path}, skipping")
            continue
        frame = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
        frame["source"] = source
        frames.append(frame)
        print(f"  loaded {source}: {frame.shape[0]} rows")

    if not frames:
        raise FileNotFoundError(
            "No raw data found. Run `python src/download_data.py` first."
        )
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce features to numeric and build the binary target."""
    feature_cols = [c for c in COLUMN_NAMES if c != "num"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary target: any disease value (1-4) -> 1, absence (0) -> 0
    df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)
    df = df.drop(columns=["num"])
    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading raw files...")
    df = load_raw()
    df = clean(df)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved cleaned dataset -> {OUTPUT_CSV}")
    print(f"  shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  target distribution:\n{df['target'].value_counts().to_string()}")
    print(f"  missing values per column:\n{df.isna().sum().to_string()}")


if __name__ == "__main__":
    main()
