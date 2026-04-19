"""Compare sentiment model results and generate a short Vietnamese summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT_DIR = "outputs/model_comparison"

MODEL_SOURCES = [
    {
        "model_name": "TF-IDF + Logistic Regression",
        "metrics_path": "evaluation/reports/tfidf_logreg_metrics.json",
        "report_path": "evaluation/reports/tfidf_logreg_classification_report.txt",
        "cost_rank": 1,
    },
    {
        "model_name": "CNN",
        "metrics_path": "evaluation/reports/cnn_metrics.json",
        "report_path": "evaluation/reports/cnn_classification_report.txt",
        "cost_rank": 2,
    },
    {
        "model_name": "LSTM/BiLSTM",
        "metrics_path": "outputs/lstm/lstm_metrics.json",
        "report_path": "outputs/lstm/lstm_classification_report.txt",
        "cost_rank": 3,
    },
    {
        "model_name": "BERT",
        "metrics_path": "outputs/bert/bert_metrics.json",
        "report_path": "outputs/bert/bert_classification_report.txt",
        "cost_rank": 4,
    },
]


def load_json(path: str | Path) -> dict:
    """Load JSON metrics if the file exists."""
    path = Path(path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_macro_avg_from_report(report_path: str | Path) -> dict[str, float | None]:
    """Extract macro precision/recall/f1 from sklearn classification_report text."""
    report_path = Path(report_path)
    if not report_path.exists():
        return {"precision": None, "recall": None, "macro_f1": None}

    text = report_path.read_text(encoding="utf-8")
    macro_line = None
    for line in text.splitlines():
        if "macro avg" in line:
            macro_line = line
            break

    if macro_line is None:
        return {"precision": None, "recall": None, "macro_f1": None}

    numbers = re.findall(r"\d+\.\d+|\d+", macro_line)
    if len(numbers) < 3:
        return {"precision": None, "recall": None, "macro_f1": None}

    return {
        "precision": float(numbers[0]),
        "recall": float(numbers[1]),
        "macro_f1": float(numbers[2]),
    }


def get_metric(metrics: dict, *keys: str) -> float | None:
    """Return the first available metric value from possible key names."""
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return None


def collect_model_results() -> pd.DataFrame:
    """Collect metrics from all configured model result files."""
    rows = []

    for source in MODEL_SOURCES:
        metrics = load_json(source["metrics_path"])
        report_metrics = parse_macro_avg_from_report(source["report_path"])

        accuracy = get_metric(metrics, "accuracy")
        precision = get_metric(metrics, "precision", "precision_macro")
        recall = get_metric(metrics, "recall", "recall_macro")
        macro_f1 = get_metric(metrics, "macro_f1", "f1_macro")
        training_time = get_metric(metrics, "training_time_seconds", "training_time")

        rows.append(
            {
                "model_name": source["model_name"],
                "accuracy": accuracy,
                "precision": precision if precision is not None else report_metrics["precision"],
                "recall": recall if recall is not None else report_metrics["recall"],
                "macro_f1": macro_f1 if macro_f1 is not None else report_metrics["macro_f1"],
                "training_time_seconds": training_time,
                "cost_rank": source["cost_rank"],
                "metrics_path": source["metrics_path"],
                "report_path": source["report_path"],
            }
        )

    return pd.DataFrame(rows)


def save_comparison_table(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Save comparison table as CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_columns = [
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "training_time_seconds",
    ]
    output_path = output_dir / "model_comparison.csv"
    df[export_columns].to_csv(output_path, index=False)
    print(f"Da luu bang so sanh: {output_path}")
    return output_path


def plot_macro_f1_comparison(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot macro F1 comparison for available models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df.dropna(subset=["macro_f1"]).sort_values("macro_f1", ascending=False)
    output_path = output_dir / "macro_f1_comparison.png"

    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_df["model_name"], plot_df["macro_f1"], color="#4c78a8")
    plt.title("Macro F1 Comparison Across Models")
    plt.xlabel("Model")
    plt.ylabel("Macro F1-score")
    plt.ylim(0, max(plot_df["macro_f1"].max() + 0.1, 1.0))
    plt.xticks(rotation=20, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Da luu bieu do macro F1: {output_path}")
    return output_path


def choose_best_model(df: pd.DataFrame) -> pd.Series:
    """Choose best model by macro F1, then accuracy."""
    available = df.dropna(subset=["macro_f1"]).copy()
    if available.empty:
        raise ValueError("Khong co macro_f1 de chon model tot nhat.")
    return available.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]


def choose_balanced_model(df: pd.DataFrame, best_macro_f1: float) -> pd.Series:
    """Choose a model balancing performance and training cost."""
    available = df.dropna(subset=["macro_f1"]).copy()
    if available.empty:
        raise ValueError("Khong co macro_f1 de chon model can bang.")

    # Prefer cheaper models within 95% of the best macro F1. If none qualify, choose
    # the cheapest model within 90%; otherwise use the best model itself.
    for threshold in [0.95, 0.90]:
        candidates = available[available["macro_f1"] >= best_macro_f1 * threshold]
        if not candidates.empty:
            return candidates.sort_values(["cost_rank", "macro_f1"], ascending=[True, False]).iloc[0]

    return choose_best_model(df)


def generate_vietnamese_commentary(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Generate and save short Vietnamese model comparison comments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best = choose_best_model(df)
    balanced = choose_balanced_model(df, best_macro_f1=float(best["macro_f1"]))

    missing_time = df["training_time_seconds"].isna().all()
    time_note = (
        "Cac file ket qua hien tai chua co day du thoi gian huan luyen, "
        "nen nhan xet chi phi dua tren do phuc tap tuong doi cua mo hinh."
        if missing_time
        else "Chi phi huan luyen duoc tham khao tu cot training_time_seconds neu co."
    )

    comment = (
        f"Mo hinh tot nhat theo macro F1 la {best['model_name']} "
        f"voi macro F1 = {best['macro_f1']:.4f} va accuracy = {best['accuracy']:.4f}. "
        f"Mo hinh can bang giua hieu nang va chi phi huan luyen la "
        f"{balanced['model_name']} voi macro F1 = {balanced['macro_f1']:.4f}. "
        f"{time_note}"
    )

    output_path = output_dir / "auto_commentary_vi.txt"
    output_path.write_text(comment, encoding="utf-8")
    print(f"Da luu nhan xet tu dong: {output_path}")
    print("\nNhan xet tu dong:")
    print(comment)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model results.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = collect_model_results()
    print("\nBang so sanh model:")
    print(
        df[
            [
                "model_name",
                "accuracy",
                "precision",
                "recall",
                "macro_f1",
                "training_time_seconds",
            ]
        ].to_string(index=False)
    )

    save_comparison_table(df, args.output_dir)
    plot_macro_f1_comparison(df, args.output_dir)
    generate_vietnamese_commentary(df, args.output_dir)


if __name__ == "__main__":
    main()
