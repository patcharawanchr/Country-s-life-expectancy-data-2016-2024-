from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path

# allow running as a standalone script
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
from load_data import load_dataset

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").T

def missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum().sort_values(ascending=False)

def corr_numeric(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    num = df.select_dtypes("number")
    if cols:
        cols = [c for c in cols if c in num.columns]
        if cols:
            num = num[cols]
    return num.corr(numeric_only=True)

def top_by_life_expectancy(df: pd.DataFrame, year: int | None = None, n: int = 10) -> pd.DataFrame:
    if "life_expectancy" not in df.columns:
        raise ValueError("Column 'life_expectancy' not found.")
    d = df.copy()
    if year is not None and "year" in d.columns:
        d = d[d["year"] == year]
    return (
        d.dropna(subset=["life_expectancy"])
         .sort_values("life_expectancy", ascending=False)
         .head(n)[["country", "year", "life_expectancy"]]
    )

def lifeexp_trend(df: pd.DataFrame, country: str) -> pd.DataFrame:
    required = {"country", "year", "life_expectancy"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns missing: {required - set(df.columns)}")
    d = (
        df.loc[df["country"].str.lower() == country.lower(), ["year", "life_expectancy"]]
          .dropna()
          .sort_values("year")
    )
    return d

def main():
    parser = argparse.ArgumentParser(description="Quick EDA helpers")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to CSV")
    parser.add_argument("--top", type=int, default=10, help="Top N by life expectancy")
    parser.add_argument("--year", type=int, default=None, help="Filter year for top list")
    parser.add_argument("--country", type=str, default=None, help="Plot trend for a country")
    parser.add_argument("--corrcols", nargs="*", help="Columns to include in correlation")
    args = parser.parse_args()

    df = load_dataset(args.csv)

    print("\n=== Missing Values ===")
    print(missing_values(df))

    print("\n=== Summary Stats (head) ===")
    print(summary_stats(df).head(15).to_string())

    print(f"\n=== Top {args.top} by Life Expectancy"
          + (f" in {args.year}" if args.year else "") + " ===")
    print(top_by_life_expectancy(df, args.year, args.top).to_string(index=False))

    print("\n=== Correlation (numeric) ===")
    print(corr_numeric(df, args.corrcols).round(3).to_string())

    if args.country:
        print(f"\n=== Trend for {args.country} ===")
        print(lifeexp_trend(df, args.country).to_string(index=False))

if __name__ == "__main__":
    main()
