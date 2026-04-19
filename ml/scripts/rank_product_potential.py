"""Rank product potential from sentiment prediction results.

Input file must contain:
- asin
- predicted_sentiment or sentiment_label

Output:
- outputs/ranking/product_ranking.csv
- outputs/ranking/top_20_product_potential.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_PATH = "data/processed/labeled_reviews.csv"
DEFAULT_OUTPUT_DIR = "outputs/ranking"
DEFAULT_OUTPUT_CSV = "outputs/ranking/product_ranking.csv"
SENTIMENT_SCORE_MAP = {
    "Negative": 1,
    "Neutral": 2,
    "Positive": 3,
}


def load_reviews(input_path: str | Path) -> pd.DataFrame:
    """Load reviews or sentiment prediction output from CSV."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Khong tim thay file input: {input_path}")
    return pd.read_csv(input_path)


def choose_sentiment_column(
    df: pd.DataFrame,
    preferred_column: str | None = None,
) -> str:
    """Choose sentiment column from predicted_sentiment or sentiment_label."""
    if preferred_column:
        if preferred_column not in df.columns:
            raise ValueError(f"Khong tim thay cot sentiment: {preferred_column}")
        return preferred_column

    for column in ["predicted_sentiment", "sentiment_label"]:
        if column in df.columns:
            return column

    raise ValueError("Input can co cot 'predicted_sentiment' hoac 'sentiment_label'.")


def validate_input_columns(df: pd.DataFrame, sentiment_col: str) -> None:
    """Validate required columns for product ranking."""
    missing_columns = [column for column in ["asin", sentiment_col] if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Input thieu cot: "
            + ", ".join(missing_columns)
            + ". Can co asin va predicted_sentiment/sentiment_label."
        )


def add_sentiment_score(
    df: pd.DataFrame,
    sentiment_col: str,
    sentiment_score_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Map sentiment labels to numeric scores."""
    sentiment_score_map = sentiment_score_map or SENTIMENT_SCORE_MAP
    scored = df.copy()
    scored[sentiment_col] = scored[sentiment_col].astype(str).str.strip()
    scored["sentiment_score"] = scored[sentiment_col].map(sentiment_score_map)

    invalid_count = scored["sentiment_score"].isna().sum()
    if invalid_count > 0:
        print(f"Bo qua {invalid_count:,} review co sentiment khong hop le.")
        scored = scored.dropna(subset=["sentiment_score"]).copy()

    scored["sentiment_score"] = scored["sentiment_score"].astype(float)
    return scored


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Normalize values to 0-1 range. Return 0 if all values are identical."""
    min_value = series.min()
    max_value = series.max()

    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.0, index=series.index)

    return (series - min_value) / (max_value - min_value)


def calculate_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate review count and sentiment ratios per asin."""
    grouped = df.groupby("asin")

    ranking = grouped.agg(
        review_count=("sentiment_score", "size"),
        sentiment_score_mean=("sentiment_score", "mean"),
        positive_count=("sentiment_score", lambda values: (values == 3).sum()),
        negative_count=("sentiment_score", lambda values: (values == 1).sum()),
    ).reset_index()

    ranking["positive_ratio"] = ranking["positive_count"] / ranking["review_count"]
    ranking["negative_ratio"] = ranking["negative_count"] / ranking["review_count"]
    return ranking


def calculate_product_potential_score(
    ranking: pd.DataFrame,
    sentiment_weight: float = 0.7,
    volume_weight: float = 0.3,
) -> pd.DataFrame:
    """Calculate product potential score with configurable weights."""
    if sentiment_weight < 0 or volume_weight < 0:
        raise ValueError("sentiment_weight va volume_weight phai >= 0.")

    total_weight = sentiment_weight + volume_weight
    if total_weight == 0:
        raise ValueError("Tong trong so phai lon hon 0.")

    sentiment_weight = sentiment_weight / total_weight
    volume_weight = volume_weight / total_weight

    ranking = ranking.copy()
    ranking["normalized_sentiment_mean"] = min_max_normalize(
        ranking["sentiment_score_mean"]
    )
    ranking["normalized_review_volume"] = min_max_normalize(ranking["review_count"])
    ranking["product_potential_score"] = (
        sentiment_weight * ranking["normalized_sentiment_mean"]
        + volume_weight * ranking["normalized_review_volume"]
    )

    return ranking.sort_values(
        ["product_potential_score", "sentiment_score_mean", "review_count"],
        ascending=False,
    ).reset_index(drop=True)


def rank_products(
    df: pd.DataFrame,
    sentiment_col: str | None = None,
    sentiment_weight: float = 0.7,
    volume_weight: float = 0.3,
    sentiment_score_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Run full product ranking pipeline."""
    sentiment_col = choose_sentiment_column(df, preferred_column=sentiment_col)
    validate_input_columns(df, sentiment_col)

    working_df = df[["asin", sentiment_col]].dropna().copy()
    working_df["asin"] = working_df["asin"].astype(str).str.strip()
    working_df = working_df[working_df["asin"] != ""]

    scored_df = add_sentiment_score(
        working_df,
        sentiment_col=sentiment_col,
        sentiment_score_map=sentiment_score_map,
    )
    ranking = calculate_product_metrics(scored_df)
    ranking = calculate_product_potential_score(
        ranking,
        sentiment_weight=sentiment_weight,
        volume_weight=volume_weight,
    )

    return ranking


def save_ranking(ranking: pd.DataFrame, output_csv: str | Path = DEFAULT_OUTPUT_CSV) -> Path:
    """Save product ranking table as CSV."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_csv, index=False)
    print(f"Da luu bang ranking: {output_csv}")
    return output_csv


def plot_top_products(
    ranking: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 20,
) -> Path:
    """Plot top products by product_potential_score."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "top_20_product_potential.png"

    top_products = ranking.head(top_n).sort_values("product_potential_score")

    plt.figure(figsize=(10, 8))
    bars = plt.barh(
        top_products["asin"],
        top_products["product_potential_score"],
        color="#4c78a8",
    )
    plt.title(f"Top {top_n} Product Potential Scores")
    plt.xlabel("Product Potential Score")
    plt.ylabel("ASIN")

    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Da luu bieu do top products: {output_path}")
    return output_path


def print_top_products(ranking: pd.DataFrame, top_n: int = 20) -> None:
    """Print top potential products."""
    columns = [
        "asin",
        "review_count",
        "sentiment_score_mean",
        "positive_ratio",
        "negative_ratio",
        "product_potential_score",
    ]
    print(f"\nTop {top_n} san pham tiem nang nhat:")
    print(ranking[columns].head(top_n).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank product potential from sentiment labels or predictions."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--sentiment-col", default=None)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sentiment-weight", type=float, default=0.7)
    parser.add_argument("--volume-weight", type=float, default=0.3)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_reviews(args.input)
    ranking = rank_products(
        df,
        sentiment_col=args.sentiment_col,
        sentiment_weight=args.sentiment_weight,
        volume_weight=args.volume_weight,
    )

    save_ranking(ranking, args.output_csv)
    plot_top_products(ranking, args.output_dir, top_n=args.top_n)
    print_top_products(ranking, top_n=args.top_n)


if __name__ == "__main__":
    main()
