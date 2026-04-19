"""Run an end-to-end demo pipeline for the Amazon sentiment project.

Default flow is laptop-friendly:
1. Load and clean reviews.
2. Create sentiment labels.
3. Preprocess text for CNN/LSTM artifacts.
4. Train or reuse TF-IDF + Logistic Regression baseline.
5. Predict one sample review.
6. Build product ranking if asin exists, or optionally create demo ASINs.
7. Print commands for backend and dashboard.

Heavy models such as CNN/LSTM/BERT should be trained separately or on Colab.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = r"C:\Users\vungo\Downloads\dataset\electronics_sample.csv"
DEFAULT_CLEAN_REVIEWS = PROJECT_ROOT / "data" / "processed" / "clean_reviews.csv"
DEFAULT_LABELED_REVIEWS = PROJECT_ROOT / "data" / "processed" / "labeled_reviews.csv"
DEFAULT_DEMO_ASIN_REVIEWS = PROJECT_ROOT / "data" / "processed" / "labeled_reviews_with_demo_asin.csv"
DEFAULT_BASELINE_MODEL = PROJECT_ROOT / "models" / "baseline" / "tfidf_logreg_model.joblib"
DEFAULT_RANKING_CSV = PROJECT_ROOT / "outputs" / "ranking" / "product_ranking.csv"


def run_command(command: list[str], step_name: str, skip: bool = False) -> None:
    """Run a subprocess command with friendly logging."""
    if skip:
        print(f"[SKIP] {step_name}")
        return

    print(f"\n[RUN] {step_name}")
    print(" ".join(str(part) for part in command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_input_exists(input_path: Path) -> None:
    """Validate input dataset path before running the pipeline."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay input dataset: {input_path}\n"
            "Hay truyen --input path/to/your_dataset.csv hoac dat file dataset vao dung duong dan."
        )


def create_demo_asin_file(input_csv: Path, output_csv: Path, num_products: int = 50) -> Path:
    """Create a demo file with synthetic ASINs if the sample data has no asin column."""
    df = pd.read_csv(input_csv)
    if "asin" in df.columns:
        print(f"[INFO] Input da co asin, dung truc tiep: {input_csv}")
        return input_csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["asin"] = [f"DEMO-ASIN-{(index % num_products) + 1:03d}" for index in range(len(df))]
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Da tao ASIN demo cho ranking: {output_csv}")
    return output_csv


def predict_sample_review(model_path: Path, review_text: str) -> None:
    """Load baseline model and predict sentiment for one review."""
    print("\n[RUN] Predict sample review")
    if not model_path.exists():
        print(f"[WARN] Chua co model baseline: {model_path}")
        return

    model = joblib.load(model_path)
    label = model.predict([review_text])[0]
    print(f"Review: {review_text}")
    print(f"Predicted label: {label}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([review_text])[0]
        for class_name, probability in zip(model.classes_, probabilities):
            print(f"  {class_name}: {probability:.4f}")


def print_next_steps() -> None:
    """Print backend/frontend commands for the live demo."""
    print("\n[DONE] Pipeline demo da hoan tat.")
    print("\nChay backend:")
    print(r".venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload")
    print("\nChay frontend:")
    print("cd frontend")
    print("npm.cmd run dev -- --host 127.0.0.1 --port 5173")
    print("\nMo dashboard:")
    print("http://127.0.0.1:5173")
    print("\nMo Swagger API:")
    print("http://127.0.0.1:8000/docs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end demo pipeline.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Raw CSV/JSONL dataset path.")
    parser.add_argument("--review-text", default="This product is great and works perfectly.")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--skip-label", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-ranking", action="store_true")
    parser.add_argument(
        "--demo-asin",
        action="store_true",
        help="Create synthetic ASINs when current labeled dataset has no asin, for dashboard demo only.",
    )
    parser.add_argument("--demo-products", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not args.skip_load:
        ensure_input_exists(input_path)

    run_command(
        [
            sys.executable,
            "ml/scripts/load_clean_reviews.py",
            "--input",
            str(input_path),
            "--output",
            str(DEFAULT_CLEAN_REVIEWS),
        ],
        "1. Load and clean data",
        skip=args.skip_load,
    )

    run_command(
        [
            sys.executable,
            "ml/scripts/create_sentiment_labels.py",
            "--input",
            str(DEFAULT_CLEAN_REVIEWS),
            "--output",
            str(DEFAULT_LABELED_REVIEWS),
        ],
        "2. Create sentiment labels",
        skip=args.skip_label,
    )

    run_command(
        [
            sys.executable,
            "ml/scripts/text_preprocessing_cnn_lstm.py",
            "--input",
            str(DEFAULT_LABELED_REVIEWS),
            "--no-nltk-download",
        ],
        "3. Preprocess text and save tokenizer/label encoder",
        skip=args.skip_preprocess,
    )

    should_skip_train = args.skip_train and DEFAULT_BASELINE_MODEL.exists()
    run_command(
        [
            sys.executable,
            "ml/scripts/train_tfidf_logreg_baseline.py",
            "--input",
            str(DEFAULT_LABELED_REVIEWS),
        ],
        "4. Train baseline model or refresh saved model",
        skip=should_skip_train,
    )

    predict_sample_review(DEFAULT_BASELINE_MODEL, args.review_text)

    ranking_input = DEFAULT_LABELED_REVIEWS
    if args.demo_asin and DEFAULT_LABELED_REVIEWS.exists():
        ranking_input = create_demo_asin_file(
            DEFAULT_LABELED_REVIEWS,
            DEFAULT_DEMO_ASIN_REVIEWS,
            num_products=args.demo_products,
        )

    run_command(
        [
            sys.executable,
            "ml/scripts/rank_product_potential.py",
            "--input",
            str(ranking_input),
            "--output-csv",
            str(DEFAULT_RANKING_CSV),
        ],
        "6. Aggregate by product and calculate ranking",
        skip=args.skip_ranking,
    )

    print_next_steps()


if __name__ == "__main__":
    main()
