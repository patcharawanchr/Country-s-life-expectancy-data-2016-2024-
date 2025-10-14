from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV = "Country's_life_expectancy_data(2016_2024).csv"

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("’", "", regex=False)
        .str.replace("'", "", regex=False)
    )
    return df

def _coerce_numeric(df: pd.DataFrame, skip: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    skip = set(skip or [])
    for c in df.columns:
        if c in skip:
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def _find_default_csv() -> Path:
    """Try to find the CSV if the exact name differs slightly."""
    p = DATA_DIR / DEFAULT_CSV
    if p.exists():
        return p
    candidates = sorted(DATA_DIR.glob("Country*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"CSV not found. Expected at {p} or a file like data/Country*.csv"
        )
    return candidates[0]

def load_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load dataset and do light cleanup:
    - drop duplicates
    - normalize column names to snake_case
    - coerce numeric where possible
    """
    path = Path(csv_path) if csv_path else _find_default_csv()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = _normalize_columns(df)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    df = _coerce_numeric(df, skip=["country", "status"])
    return df

if __name__ == "__main__":
    d = load_dataset()
    print("Rows:", len(d), "| Columns:", list(d.columns))
    print(d.head(10).to_string(index=False))
