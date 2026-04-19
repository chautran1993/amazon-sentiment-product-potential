"""EDA for labeled Amazon Electronics reviews.

Script nay doc data/processed/labeled_reviews.csv, tao cac thong ke EDA co ban
va luu tung chart rieng vao outputs/eda/.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_PATH = "data/processed/labeled_reviews.csv"
DEFAULT_OUTPUT_DIR = "outputs/eda"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


def load_labeled_reviews(input_path: str | Path) -> pd.DataFrame:
    """Load labeled reviews and validate columns needed for EDA."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Khong tim thay file input: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = ["reviewText", "overall", "sentiment_label"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError("Dataset thieu cot: " + ", ".join(missing_columns))

    df["reviewText"] = df["reviewText"].fillna("").astype(str)
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df = df.dropna(subset=["overall", "sentiment_label"])
    df["overall"] = df["overall"].astype(int)
    return df


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create output directory for all EDA charts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_current_plot(output_path: Path) -> None:
    """Save current matplotlib figure and close it to avoid overlapping plots."""
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Da luu chart: {output_path}")


def plot_rating_distribution(df: pd.DataFrame, output_dir: Path) -> pd.Series:
    """Rating distribution shows whether reviews lean toward low or high scores."""
    rating_counts = df["overall"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(rating_counts.index.astype(str), rating_counts.values, color="#4c78a8")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    save_current_plot(output_dir / "01_rating_distribution.png")

    return rating_counts


def plot_sentiment_distribution(df: pd.DataFrame, output_dir: Path) -> pd.Series:
    """Sentiment distribution summarizes class counts for Negative/Neutral/Positive."""
    label_order = ["Negative", "Neutral", "Positive"]
    label_counts = df["sentiment_label"].value_counts().reindex(label_order, fill_value=0)

    plt.figure(figsize=(8, 5))
    plt.bar(label_counts.index, label_counts.values, color=["#d95f59", "#6b7280", "#2f9e6d"])
    plt.title("Sentiment Label Distribution")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Number of Reviews")
    save_current_plot(output_dir / "02_sentiment_label_distribution.png")

    return label_counts


def plot_top_reviewed_products(df: pd.DataFrame, output_dir: Path) -> pd.Series | None:
    """Top reviewed products identify items with the most customer feedback."""
    if "asin" not in df.columns:
        print("Bo qua top products: dataset khong co cot asin.")
        return None

    top_products = df["asin"].value_counts().head(20).sort_values()

    plt.figure(figsize=(10, 8))
    plt.barh(top_products.index.astype(str), top_products.values, color="#59a14f")
    plt.title("Top 20 Products by Number of Reviews")
    plt.xlabel("Number of Reviews")
    plt.ylabel("ASIN")
    save_current_plot(output_dir / "03_top_20_products_by_reviews.png")

    return top_products.sort_values(ascending=False)


def add_review_length(df: pd.DataFrame) -> pd.DataFrame:
    """Review length helps inspect whether text is too short or very long."""
    df = df.copy()
    df["review_word_count"] = df["reviewText"].str.split().str.len()
    return df


def plot_review_length_distribution(df: pd.DataFrame, output_dir: Path) -> pd.Series:
    """Review word-count distribution shows the typical amount of text per review."""
    df = add_review_length(df)

    plt.figure(figsize=(9, 5))
    plt.hist(df["review_word_count"], bins=50, color="#f28e2b", edgecolor="black")
    plt.title("Review Length Distribution by Word Count")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Reviews")
    save_current_plot(output_dir / "04_review_length_distribution.png")

    return df["review_word_count"].describe()


def tokenize_review_text(text: str) -> list[str]:
    """Basic text cleaning: lowercase, keep alphabetic words, remove stopwords."""
    words = re.findall(r"[a-z]+", text.lower())
    return [word for word in words if word not in STOPWORDS and len(word) > 2]


def get_top_words(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Top words reveal frequent topics, product terms, or common review language."""
    counter: Counter[str] = Counter()
    for text in df["reviewText"]:
        counter.update(tokenize_review_text(text))

    return pd.DataFrame(counter.most_common(top_n), columns=["word", "count"])


def plot_top_words(df: pd.DataFrame, output_dir: Path, top_n: int = 30) -> pd.DataFrame:
    """Top cleaned words help describe common themes in the review corpus."""
    top_words = get_top_words(df, top_n=top_n)
    plot_data = top_words.sort_values("count")

    plt.figure(figsize=(10, 8))
    plt.barh(plot_data["word"], plot_data["count"], color="#b07aa1")
    plt.title(f"Top {top_n} Words After Basic Cleaning")
    plt.xlabel("Word Count")
    plt.ylabel("Word")
    save_current_plot(output_dir / "05_top_words_basic_cleaning.png")

    return top_words


def analyze_class_imbalance(label_counts: pd.Series, output_dir: Path) -> dict[str, float]:
    """Class imbalance check shows if one sentiment class dominates the dataset."""
    total = label_counts.sum()
    percentages = (label_counts / total * 100).round(2)
    majority_label = label_counts.idxmax()
    minority_label = label_counts.idxmin()
    imbalance_ratio = round(label_counts.max() / max(label_counts.min(), 1), 2)

    plt.figure(figsize=(8, 5))
    plt.bar(percentages.index, percentages.values, color=["#d95f59", "#6b7280", "#2f9e6d"])
    plt.title("Sentiment Class Balance Percentage")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Percentage of Reviews")
    save_current_plot(output_dir / "06_class_imbalance_percentage.png")

    return {
        "total_reviews": float(total),
        "majority_class_count": float(label_counts.max()),
        "minority_class_count": float(label_counts.min()),
        "imbalance_ratio": imbalance_ratio,
        "majority_label": majority_label,
        "minority_label": minority_label,
    }


def print_report(
    rating_counts: pd.Series,
    label_counts: pd.Series,
    top_products: pd.Series | None,
    length_stats: pd.Series,
    top_words: pd.DataFrame,
    imbalance_stats: dict[str, float],
) -> None:
    """Print concise EDA results for quick reporting."""
    print("\n=== 1. Phan bo rating ===")
    print("Y nghia: cho biet review tap trung nhieu o muc sao nao.")
    print(rating_counts.to_string())

    print("\n=== 2. Phan bo sentiment_label ===")
    print("Y nghia: cho biet so mau cua tung lop sentiment.")
    print(label_counts.to_string())

    print("\n=== 3. Top 20 san pham co nhieu review nhat ===")
    print("Y nghia: cac san pham nay co nhieu du lieu hon de phan tich tiem nang.")
    if top_products is None:
        print("Khong co cot asin nen khong tinh duoc top san pham.")
    else:
        print(top_products.to_string())

    print("\n=== 4. Phan bo do dai review theo so tu ===")
    print("Y nghia: giup phat hien review qua ngan/qua dai va quyet dinh nguong loc text.")
    print(length_stats.to_string())

    print("\n=== 5. Top tu xuat hien nhieu nhat sau khi lam sach co ban ===")
    print("Y nghia: goi y chu de/noi dung thuong gap trong review.")
    print(top_words.to_string(index=False))

    print("\n=== 6. Kiem tra mat can bang du lieu ===")
    print("Y nghia: neu imbalance_ratio cao, can can nhac class weight/resampling khi train.")
    print(f"Majority class: {imbalance_stats['majority_label']}")
    print(f"Minority class: {imbalance_stats['minority_label']}")
    print(f"Imbalance ratio: {imbalance_stats['imbalance_ratio']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA for labeled Amazon reviews.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Duong dan file labeled_reviews.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Thu muc luu cac chart EDA.",
    )
    parser.add_argument(
        "--top-words",
        type=int,
        default=30,
        help="So tu pho bien nhat can ve bieu do.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    df = load_labeled_reviews(args.input)
    rating_counts = plot_rating_distribution(df, output_dir)
    label_counts = plot_sentiment_distribution(df, output_dir)
    top_products = plot_top_reviewed_products(df, output_dir)
    length_stats = plot_review_length_distribution(df, output_dir)
    top_words = plot_top_words(df, output_dir, top_n=args.top_words)
    imbalance_stats = analyze_class_imbalance(label_counts, output_dir)

    print_report(
        rating_counts=rating_counts,
        label_counts=label_counts,
        top_products=top_products,
        length_stats=length_stats,
        top_words=top_words,
        imbalance_stats=imbalance_stats,
    )


if __name__ == "__main__":
    main()
