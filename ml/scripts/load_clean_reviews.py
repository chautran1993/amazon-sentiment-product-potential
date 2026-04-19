"""Load and clean Amazon Reviews 2018 Electronics data.

Script nay ho tro input CSV hoac JSON Lines (.jsonl/.json.gz) va luu file
da lam sach vao data/processed/clean_reviews.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["reviewText", "overall"]
KEEP_COLUMNS = [
    "reviewerID",
    "asin",
    "reviewText",
    "overall",
    "summary",
    "unixReviewTime",
]


def get_available_columns(input_path: Path) -> list[str] | None:
    """Doc nhanh header CSV de chi load cac cot can thiet neu co the."""
    suffixes = "".join(input_path.suffixes).lower()

    if suffixes.endswith(".csv"):
        header = pd.read_csv(input_path, nrows=0)
        return [column for column in KEEP_COLUMNS if column in header.columns]

    # JSON lines khong co header rieng, pandas se tu suy ra columns khi doc.
    return None


def load_reviews(input_path: str | Path) -> pd.DataFrame:
    """Load dataset tu CSV hoac JSON Lines bang pandas."""
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Khong tim thay file input: {input_path}")

    suffixes = "".join(input_path.suffixes).lower()
    available_columns = get_available_columns(input_path)

    if suffixes.endswith(".csv"):
        if not available_columns:
            raise ValueError("CSV khong co cot nao nam trong danh sach can giu.")
        return pd.read_csv(input_path, usecols=available_columns)

    if (
        suffixes.endswith(".json")
        or suffixes.endswith(".jsonl")
        or suffixes.endswith(".json.gz")
        or suffixes.endswith(".jsonl.gz")
    ):
        df = pd.read_json(input_path, lines=True)
        available_columns = [column for column in KEEP_COLUMNS if column in df.columns]
        if not available_columns:
            raise ValueError("JSON Lines khong co cot nao nam trong danh sach can giu.")
        return df[available_columns]

    raise ValueError("Chi ho tro file .csv, .json, .jsonl, .json.gz, .jsonl.gz")


def validate_required_columns(df: pd.DataFrame) -> None:
    """Dam bao cac cot bat buoc de clean sentiment review ton tai."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset thieu cot bat buoc: "
            + ", ".join(missing_columns)
            + ". Can co reviewText va overall."
        )


def count_words(text: object) -> int:
    """Dem so tu trong review text."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def clean_reviews(df: pd.DataFrame, min_words: int = 5) -> pd.DataFrame:
    """Lam sach review: null, duplicate, rating loi va review qua ngan."""
    validate_required_columns(df)

    cleaned = df.copy()

    # Chuan hoa text va rating truoc khi loc null.
    cleaned["reviewText"] = cleaned["reviewText"].astype("string").str.strip()
    cleaned["overall"] = pd.to_numeric(cleaned["overall"], errors="coerce")

    # Loai bo review thieu text/rating.
    cleaned = cleaned.dropna(subset=["reviewText", "overall"])
    cleaned = cleaned[cleaned["reviewText"] != ""]

    # Loai duplicate tren cac cot dang giu trong dataframe.
    cleaned = cleaned.drop_duplicates()

    # Loai review qua ngan, vi thuong khong du thong tin cho sentiment analysis.
    cleaned = cleaned[cleaned["reviewText"].map(count_words) >= min_words]

    return cleaned.reset_index(drop=True)


def print_dataset_report(initial_rows: int, cleaned: pd.DataFrame) -> None:
    """In thong ke sau khi load va clean."""
    print(f"So luong dong ban dau: {initial_rows:,}")
    print(f"So luong dong sau khi lam sach: {len(cleaned):,}")

    if "asin" in cleaned.columns:
        print(f"So luong unique products theo asin: {cleaned['asin'].nunique():,}")
    else:
        print("So luong unique products theo asin: khong co cot asin trong dataset")

    print("Phan bo rating:")
    rating_distribution = cleaned["overall"].value_counts().sort_index()
    print(rating_distribution.to_string())


def save_clean_reviews(df: pd.DataFrame, output_path: str | Path) -> None:
    """Luu dataset da lam sach ra CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Da luu file clean: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load va clean Amazon Reviews 2018 category Electronics."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Duong dan file input .csv hoac JSON Lines .json/.jsonl/.json.gz",
    )
    parser.add_argument(
        "--output",
        default="data/processed/clean_reviews.csv",
        help="Duong dan file CSV output sau khi lam sach.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=5,
        help="So tu toi thieu trong reviewText.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_reviews = load_reviews(args.input)
    initial_rows = len(raw_reviews)

    cleaned_reviews = clean_reviews(raw_reviews, min_words=args.min_words)
    print_dataset_report(initial_rows, cleaned_reviews)
    save_clean_reviews(cleaned_reviews, args.output)


if __name__ == "__main__":
    main()
