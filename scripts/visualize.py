from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# allow running as a standalone script
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
from load_data import load_dataset

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def plot_lifeexp_distribution(df: pd.DataFrame, save: bool = True):
    if "life_expectancy" not in df.columns:
        raise ValueError("Column 'life_expectancy' not found.")
    df["life_expectancy"].plot(kind="hist", bins=30)
    plt.title("Distribution of Life Expectancy (2016–2024)")
    plt.xlabel("Life Expectancy (years)")
    plt.ylabel("Count")
    plt.tight_layout()
    if save:
        path = OUT_DIR / "life_expectancy_hist.png"
        plt.savefig(path, dpi=200)
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.clf()

def plot_gdp_vs_lifeexp(df: pd.DataFrame, logx: bool = True, save: bool = True):
    required = {"gdp", "life_expectancy"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns missing: {required - set(df.columns)}")
    d = df.dropna(subset=["gdp", "life_expectancy"])
    ax = d.plot(kind="scatter", x="gdp", y="life_expectancy", alpha=0.5)
    ax.set_title("GDP vs Life Expectancy")
    ax.set_xlabel("GDP")
    ax.set_ylabel("Life Expectancy (years)")
    if logx:
        ax.set_xscale("log")
        ax.set_xlabel("GDP (log scale)")
    plt.tight_layout()
    if save:
        path = OUT_DIR / "gdp_vs_life_expectancy.png"
        plt.savefig(path, dpi=200)
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.clf()

def plot_top_countries_by_year(df: pd.DataFrame, year: int, n: int = 10, save: bool = True):
    required = {"country", "year", "life_expectancy"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns missing: {required - set(df.columns)}")
    d = (
        df[df["year"] == year]
        .dropna(subset=["life_expectancy"])
        .sort_values("life_expectancy", ascending=False)
        .head(n)
    )
    ax = d.plot(kind="bar", x="country", y="life_expectancy", rot=45)
    ax.set_title(f"Top {n} Countries by Life Expectancy in {year}")
    ax.set_xlabel("Country")
    ax.set_ylabel("Life Expectancy (years)")
    plt.tight_layout()
    if save:
        path = OUT_DIR / f"top_{n}_life_expectancy_{year}.png"
        plt.savefig(path, dpi=200)
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.clf()

def plot_country_trend(df: pd.DataFrame, country: str, save: bool = True):
    required = {"country", "year", "life_expectancy"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns missing: {required - set(df.columns)}")
    d = (
        df.loc[df["country"].str.lower() == country.lower(), ["year", "life_expectancy"]]
          .dropna()
          .sort_values("year")
    )
    d.plot(x="year", y="life_expectancy", marker="o")
    plt.title(f"Life Expectancy Trend: {country}")
    plt.xlabel("Year")
    plt.ylabel("Life Expectancy (years)")
    plt.tight_layout()
    if save:
        safe = country.lower().replace(" ", "_")
        path = OUT_DIR / f"trend_{safe}.png"
        plt.savefig(path, dpi=200)
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.clf()

def _demo():
    df = load_dataset()
    plot_lifeexp_distribution(df, save=True)
    if "gdp" in df.columns:
        plot_gdp_vs_lifeexp(df, logx=True, save=True)
    if "year" in df.columns:
        try:
            year = int(df["year"].dropna().astype(int).mode().iloc[0])
        except Exception:
            year = int(df["year"].dropna().astype(int).max())
        plot_top_countries_by_year(df, year=year, n=10, save=True)
    plot_country_trend(df, country="Thailand", save=True)

if __name__ == "__main__":
    _demo()
