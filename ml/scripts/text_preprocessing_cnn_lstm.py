"""Reusable text preprocessing utilities for CNN/LSTM sentiment models.

Pipeline:
1. Clean raw review text.
2. Tokenize with NLTK and remove English stopwords.
3. Fit a Keras tokenizer and convert text to padded sequences.
4. Encode sentiment labels.
5. Split train/validation/test.
6. Compute class weights for imbalanced data.
7. Save tokenizer and label encoder to artifacts/.
"""

from __future__ import annotations

import argparse
import html
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from nltk import data as nltk_data
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
except ImportError as exc:
    raise ImportError(
        "Can cai TensorFlow de dung Keras Tokenizer: pip install tensorflow"
    ) from exc


DEFAULT_INPUT_PATH = "data/processed/labeled_reviews.csv"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_MAX_LEN = 256
DEFAULT_NUM_WORDS = 50_000
LABEL_ORDER = ["Negative", "Neutral", "Positive"]


@dataclass
class PreprocessingResult:
    """Container returned by the full preprocessing pipeline."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    tokenizer: Any
    label_encoder: LabelEncoder
    class_weights: dict[int, float]
    cleaned_texts: pd.Series


def ensure_nltk_resources(download_if_missing: bool = True) -> None:
    """Ensure NLTK resources for tokenization and English stopwords are available."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
    ]

    # Newer NLTK versions may require punkt_tab for word_tokenize.
    optional_resources = [("tokenizers/punkt_tab", "punkt_tab")]

    for resource_path, package_name in resources + optional_resources:
        try:
            nltk_data.find(resource_path)
        except LookupError:
            if download_if_missing:
                nltk_download(package_name)
            elif resource_path in dict(resources):
                raise


def remove_html_tags(text: str) -> str:
    """Remove HTML tags and unescape HTML entities."""
    text = html.unescape(text)
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove http, https, and www URLs from review text."""
    return re.sub(r"(https?://\S+|www\.\S+)", " ", text)


def clean_text_basic(text: object) -> str:
    """Lowercase, remove HTML/URLs/special characters, and normalize spaces."""
    if pd.isna(text):
        return ""

    cleaned = str(text).lower()
    cleaned = remove_html_tags(cleaned)
    cleaned = remove_urls(cleaned)
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize_without_stopwords(text: str, english_stopwords: set[str]) -> list[str]:
    """Tokenize with NLTK and remove English stopwords."""
    tokens = word_tokenize(text)
    return [
        token
        for token in tokens
        if token.isalpha() and token not in english_stopwords and len(token) > 1
    ]


def preprocess_review_texts(
    texts: pd.Series,
    download_nltk_if_missing: bool = True,
) -> pd.Series:
    """Clean, tokenize, remove stopwords, then join tokens for Keras Tokenizer."""
    ensure_nltk_resources(download_if_missing=download_nltk_if_missing)
    english_stopwords = set(stopwords.words("english"))

    processed_texts = []
    for text in texts:
        cleaned = clean_text_basic(text)
        tokens = tokenize_without_stopwords(cleaned, english_stopwords)
        processed_texts.append(" ".join(tokens))

    return pd.Series(processed_texts, index=texts.index, name="clean_text")


def create_keras_tokenizer(
    texts: pd.Series | list[str],
    num_words: int = DEFAULT_NUM_WORDS,
    oov_token: str = "<OOV>",
) -> Any:
    """Fit a Keras tokenizer on preprocessed training text."""
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded_sequences(
    tokenizer: Any,
    texts: pd.Series | list[str],
    max_len: int = DEFAULT_MAX_LEN,
) -> np.ndarray:
    """Convert text to integer sequences and pad/truncate to max_len."""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(
        sequences,
        maxlen=max_len,
        padding="post",
        truncating="post",
    )


def encode_sentiment_labels(labels: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode Negative/Neutral/Positive labels into numeric values."""
    label_encoder = LabelEncoder()
    label_encoder.fit(LABEL_ORDER)

    unknown_labels = sorted(set(labels.dropna()) - set(LABEL_ORDER))
    if unknown_labels:
        raise ValueError("Nhan sentiment khong hop le: " + ", ".join(unknown_labels))

    encoded_labels = label_encoder.transform(labels)
    return encoded_labels, label_encoder


def can_stratify(labels: np.ndarray) -> bool:
    """Check if every class has enough samples for stratified splitting."""
    _, counts = np.unique(labels, return_counts=True)
    return len(counts) > 1 and counts.min() >= 2


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train, validation, and test sets."""
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size va val_size phai > 0 va tong nho hon 1.")

    temp_size = test_size + val_size
    stratify_main = y if can_stratify(y) else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=random_state,
        stratify=stratify_main,
    )

    relative_test_size = test_size / temp_size
    stratify_temp = y_temp if can_stratify(y_temp) else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=stratify_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Compute class weights to reduce the effect of class imbalance."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


def save_preprocessing_artifacts(
    tokenizer: Any,
    label_encoder: LabelEncoder,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> None:
    """Save tokenizer and label encoder for inference/reuse."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with (artifacts_dir / "keras_tokenizer.pkl").open("wb") as file:
        pickle.dump(tokenizer, file)

    joblib.dump(label_encoder, artifacts_dir / "label_encoder.joblib")
    print(f"Da luu tokenizer: {artifacts_dir / 'keras_tokenizer.pkl'}")
    print(f"Da luu label encoder: {artifacts_dir / 'label_encoder.joblib'}")


def prepare_cnn_lstm_data(
    df: pd.DataFrame,
    text_col: str = "reviewText",
    label_col: str = "sentiment_label",
    max_len: int = DEFAULT_MAX_LEN,
    num_words: int = DEFAULT_NUM_WORDS,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    save_artifacts: bool = True,
    download_nltk_if_missing: bool = True,
) -> PreprocessingResult:
    """Run the full reusable preprocessing pipeline for CNN/LSTM models."""
    missing_columns = [column for column in [text_col, label_col] if column not in df.columns]
    if missing_columns:
        raise ValueError("Dataset thieu cot: " + ", ".join(missing_columns))

    working_df = df[[text_col, label_col]].dropna().copy()
    working_df[label_col] = working_df[label_col].astype(str).str.strip()
    cleaned_texts = preprocess_review_texts(
        working_df[text_col],
        download_nltk_if_missing=download_nltk_if_missing,
    )
    non_empty_mask = cleaned_texts.str.len() > 0
    working_df = working_df.loc[non_empty_mask].copy()
    cleaned_texts = cleaned_texts.loc[non_empty_mask]
    encoded_labels, label_encoder = encode_sentiment_labels(working_df[label_col])

    X_text_train, X_text_temp, y_train, y_temp = train_test_split(
        cleaned_texts,
        encoded_labels,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=encoded_labels if can_stratify(encoded_labels) else None,
    )

    relative_test_size = test_size / (test_size + val_size)
    X_text_val, X_text_test, y_val, y_test = train_test_split(
        X_text_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp if can_stratify(y_temp) else None,
    )

    tokenizer = create_keras_tokenizer(X_text_train, num_words=num_words)
    X_train = texts_to_padded_sequences(tokenizer, X_text_train, max_len=max_len)
    X_val = texts_to_padded_sequences(tokenizer, X_text_val, max_len=max_len)
    X_test = texts_to_padded_sequences(tokenizer, X_text_test, max_len=max_len)
    class_weights = get_class_weights(y_train)

    if save_artifacts:
        save_preprocessing_artifacts(tokenizer, label_encoder, artifacts_dir)

    return PreprocessingResult(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        class_weights=class_weights,
        cleaned_texts=cleaned_texts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess review text for CNN/LSTM sentiment classification."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--num-words", type=int, default=DEFAULT_NUM_WORDS)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-nltk-download",
        action="store_true",
        help="Khong tu dong download punkt/stopwords neu NLTK resource bi thieu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    result = prepare_cnn_lstm_data(
        df=df,
        max_len=args.max_len,
        num_words=args.num_words,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        artifacts_dir=args.artifacts_dir,
        save_artifacts=True,
        download_nltk_if_missing=not args.no_nltk_download,
    )

    print("Preprocessing hoan tat.")
    print(f"X_train: {result.X_train.shape}, y_train: {result.y_train.shape}")
    print(f"X_val:   {result.X_val.shape}, y_val:   {result.y_val.shape}")
    print(f"X_test:  {result.X_test.shape}, y_test:  {result.y_test.shape}")
    print(f"Label classes: {list(result.label_encoder.classes_)}")
    print(f"Class weights: {result.class_weights}")


if __name__ == "__main__":
    main()
