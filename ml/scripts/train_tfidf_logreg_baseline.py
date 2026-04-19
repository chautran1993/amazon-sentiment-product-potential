"""Train a TF-IDF + Logistic Regression baseline for 3-class sentiment.

Baseline nay huu ich vi no nhanh, de giai thich va tao moc so sanh ro rang
truoc khi dung CNN/LSTM hoac cac mo hinh deep learning nang hon.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_INPUT_PATH = "data/processed/labeled_reviews.csv"
DEFAULT_CONFUSION_MATRIX_PATH = "evaluation/figures/tfidf_logreg_confusion_matrix.png"
DEFAULT_REPORTS_DIR = "evaluation/reports"
DEFAULT_MODEL_PATH = "models/baseline/tfidf_logreg_model.joblib"
RANDOM_STATE = 42
LABEL_ORDER = ["Negative", "Neutral", "Positive"]


def load_dataset(input_path: str | Path) -> pd.DataFrame:
    """Load labeled reviews and keep rows needed for supervised training."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Khong tim thay file input: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = ["reviewText", "sentiment_label"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError("Dataset thieu cot: " + ", ".join(missing_columns))

    df = df[required_columns].dropna().copy()
    df["reviewText"] = df["reviewText"].astype(str).str.strip()
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip()
    df = df[df["reviewText"] != ""]
    df = df[df["sentiment_label"].isin(LABEL_ORDER)]

    if df.empty:
        raise ValueError("Dataset rong sau khi loc reviewText va sentiment_label.")

    return df.reset_index(drop=True)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split data into train/test with fixed random_state and stratification."""
    return train_test_split(
        df["reviewText"],
        df["sentiment_label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["sentiment_label"],
    )


def build_tfidf_logreg_model(random_state: int = RANDOM_STATE) -> Pipeline:
    """Create TF-IDF + Logistic Regression baseline pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_model(y_true: pd.Series, y_pred: list[str]) -> dict[str, float]:
    """Compute and print standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=0,
        ),
    }

    print("\n=== Baseline Metrics ===")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision macro: {metrics['precision_macro']:.4f}")
    print(f"Recall macro:    {metrics['recall_macro']:.4f}")
    print(f"Macro F1-score:  {metrics['macro_f1']:.4f}")

    print("\n=== Confusion Matrix ===")
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    print(pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER).to_string())

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            zero_division=0,
        )
    )

    return metrics


def save_evaluation_outputs(
    y_true: pd.Series,
    y_pred: list[str],
    metrics: dict[str, float],
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> None:
    """Save baseline metrics and classification report for model comparison."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    metrics_path = reports_dir / "tfidf_logreg_metrics.json"
    report_path = reports_dir / "tfidf_logreg_classification_report.txt"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with report_path.open("w", encoding="utf-8") as file:
        file.write(report)

    print(f"Da luu metrics: {metrics_path}")
    print(f"Da luu classification report: {report_path}")


def save_model(model: Pipeline, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
    """Save trained sklearn pipeline for FastAPI prediction demo."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Da luu model baseline: {model_path}")


def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: list[str],
    output_path: str | Path = DEFAULT_CONFUSION_MATRIX_PATH,
) -> None:
    """Save confusion matrix as a matplotlib image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("TF-IDF + Logistic Regression Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(LABEL_ORDER)), LABEL_ORDER, rotation=45, ha="right")
    plt.yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    max_value = matrix.max()
    threshold = max_value / 2 if max_value else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            color = "white" if value > threshold else "black"
            plt.text(col_idx, row_idx, str(value), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Da luu confusion matrix: {output_path}")


def print_baseline_explanation() -> None:
    """Explain why this baseline is useful before deep learning."""
    print(
        "\nBaseline TF-IDF + Logistic Regression huu ich vi no nhanh, de chay, "
        "de giai thich va tao moc so sanh ban dau. Neu CNN/LSTM khong vuot baseline "
        "nay mot cach ro rang, nhom can xem lai preprocessing, data split hoac kien truc model."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression baseline for sentiment classification."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--confusion-matrix", default=DEFAULT_CONFUSION_MATRIX_PATH)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataset(args.input)
    X_train, X_test, y_train, y_test = split_train_test(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"So dong train: {len(X_train):,}")
    print(f"So dong test:  {len(X_test):,}")

    model = build_tfidf_logreg_model(random_state=args.random_state)
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    metrics["training_time_seconds"] = training_time
    save_confusion_matrix_plot(y_test, y_pred, args.confusion_matrix)
    save_evaluation_outputs(y_test, y_pred, metrics, args.reports_dir)
    save_model(model, args.model_path)
    print_baseline_explanation()


if __name__ == "__main__":
    main()
