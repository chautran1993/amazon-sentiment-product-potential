"""Create sentiment labels from cleaned Amazon review ratings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_PATH = "data/processed/clean_reviews.csv"
DEFAULT_OUTPUT_PATH = "data/processed/labeled_reviews.csv"
DEFAULT_PLOT_PATH = "evaluation/figures/sentiment_label_distribution.png"


def load_clean_reviews(input_path: str | Path) -> pd.DataFrame:
    """Load cleaned reviews from CSV."""
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Khong tim thay file input: {input_path}")

    df = pd.read_csv(input_path)
    if "overall" not in df.columns:
        raise ValueError("Dataset can co cot 'overall' de tao sentiment_label.")

    return df


def rating_to_sentiment_label(rating: float) -> str:
    """Map Amazon rating to sentiment label."""
    if rating in (1, 2):
        return "Negative"
    if rating == 3:
        return "Neutral"
    if rating in (4, 5):
        return "Positive"
    raise ValueError(f"Rating khong hop le: {rating}")


def add_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment_label column based on overall rating."""
    labeled = df.copy()
    labeled["overall"] = pd.to_numeric(labeled["overall"], errors="coerce")
    labeled = labeled.dropna(subset=["overall"])
    labeled["overall"] = labeled["overall"].astype(int)

    valid_ratings = labeled["overall"].between(1, 5)
    invalid_count = len(labeled) - valid_ratings.sum()
    if invalid_count > 0:
        print(f"Bo qua {invalid_count:,} dong co rating ngoai khoang 1-5.")
        labeled = labeled[valid_ratings].copy()

    labeled["sentiment_label"] = labeled["overall"].map(rating_to_sentiment_label)
    return labeled.reset_index(drop=True)


def get_label_distribution(df: pd.DataFrame) -> pd.Series:
    """Return label counts in a stable display order."""
    label_order = ["Negative", "Neutral", "Positive"]
    return df["sentiment_label"].value_counts().reindex(label_order, fill_value=0)


def print_label_distribution(label_counts: pd.Series) -> None:
    """Print sentiment label distribution."""
    print("Phan bo sentiment_label:")
    print(label_counts.to_string())


def plot_label_distribution(label_counts: pd.Series, plot_path: str | Path) -> None:
    """Plot and save sentiment label distribution as a bar chart."""
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(label_counts.index, label_counts.values, color=["#d95f59", "#6b7280", "#2f9e6d"])
    plt.title("Sentiment Label Distribution")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Number of Reviews")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Da luu bieu do: {plot_path}")


def save_labeled_reviews(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save labeled reviews to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Da luu file labeled: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create sentiment_label from Amazon review ratings."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Duong dan file clean_reviews.csv.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Duong dan file labeled_reviews.csv.",
    )
    parser.add_argument(
        "--plot",
        default=DEFAULT_PLOT_PATH,
        help="Duong dan luu bieu do phan bo nhan sentiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    clean_reviews = load_clean_reviews(args.input)
    labeled_reviews = add_sentiment_labels(clean_reviews)
    label_counts = get_label_distribution(labeled_reviews)

    print_label_distribution(label_counts)
    plot_label_distribution(label_counts, args.plot)
    save_labeled_reviews(labeled_reviews, args.output)


if __name__ == "__main__":
    main()
