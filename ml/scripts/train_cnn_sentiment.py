"""Train a CNN sentiment classifier with TensorFlow/Keras.

This script reuses the existing CNN/LSTM preprocessing module, trains a
production-friendly 1D CNN, evaluates it on the test set, and saves the best
model plus reporting artifacts.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

try:
    from text_preprocessing_cnn_lstm import (
        DEFAULT_MAX_LEN,
        DEFAULT_NUM_WORDS,
        LABEL_ORDER,
        prepare_cnn_lstm_data,
    )
except ImportError:
    from ml.scripts.text_preprocessing_cnn_lstm import (
        DEFAULT_MAX_LEN,
        DEFAULT_NUM_WORDS,
        LABEL_ORDER,
        prepare_cnn_lstm_data,
    )


DEFAULT_INPUT_PATH = "data/processed/labeled_reviews.csv"
DEFAULT_MODEL_DIR = "models/cnn"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_FIGURES_DIR = "evaluation/figures"
DEFAULT_REPORTS_DIR = "evaluation/reports"
RANDOM_STATE = 42


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    """Make model training more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(
    vocab_size: int,
    max_len: int = DEFAULT_MAX_LEN,
    num_classes: int = 3,
    embedding_dim: int = 128,
    filters: int = 128,
    kernel_size: int = 5,
    dense_units: int = 64,
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build and compile a 1D CNN sentiment classification model."""
    model = Sequential(
        [
            Input(shape=(max_len,), dtype="int32", name="input_tokens"),
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="embedding",
            ),
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                name="conv1d",
            ),
            GlobalMaxPooling1D(name="global_max_pooling"),
            Dense(dense_units, activation="relu", name="dense_hidden"),
            Dropout(dropout_rate, name="dropout"),
            Dense(num_classes, activation="softmax", name="sentiment_output"),
        ],
        name="cnn_sentiment_classifier",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    epochs: int = 10,
    batch_size: int = 64,
    patience: int = 3,
) -> tf.keras.callbacks.History:
    """Train model with EarlyStopping and ModelCheckpoint."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best_model.weights.h5"
    model_config_path = model_dir / "model_config.json"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    if checkpoint_path.exists():
        model.load_weights(checkpoint_path)

    with model_config_path.open("w", encoding="utf-8") as file:
        file.write(model.to_json())

    print(f"Best weights checkpoint: {checkpoint_path}")
    print(f"Model architecture saved: {model_config_path}")
    return history


def load_best_model(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Load model architecture and best weights from models/cnn/."""
    model_dir = Path(model_dir)
    model_config_path = model_dir / "model_config.json"
    checkpoint_path = model_dir / "best_model.weights.h5"

    if not model_config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            "Khong tim thay model_config.json hoac best_model.weights.h5 trong "
            f"{model_dir}"
        )

    model = model_from_json(model_config_path.read_text(encoding="utf-8"))
    model.load_weights(checkpoint_path)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(
    history: tf.keras.callbacks.History,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Plot train/validation loss and accuracy as separate figures."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title("CNN Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = figures_dir / "cnn_training_loss.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Da luu loss plot: {loss_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title("CNN Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    accuracy_path = figures_dir / "cnn_training_accuracy.png"
    plt.savefig(accuracy_path, dpi=150)
    plt.close()
    print(f"Da luu accuracy plot: {accuracy_path}")


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> Path:
    """Save confusion matrix image for the CNN model."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "cnn_confusion_matrix.png"

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_ORDER))))

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("CNN Confusion Matrix")
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
    return output_path


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> dict[str, float]:
    """Evaluate model with accuracy, macro F1, confusion matrix, and report."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=list(range(len(LABEL_ORDER))))
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(LABEL_ORDER))),
        target_names=LABEL_ORDER,
        zero_division=0,
    )

    print("\n=== CNN Test Metrics ===")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\n=== Confusion Matrix ===")
    print(pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER).to_string())

    print("\n=== Classification Report ===")
    print(report)

    save_confusion_matrix_plot(y_test, y_pred, figures_dir=figures_dir)

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }
    metrics_path = reports_dir / "cnn_metrics.json"
    report_path = reports_dir / "cnn_classification_report.txt"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with report_path.open("w", encoding="utf-8") as file:
        file.write(report)

    print(f"Da luu metrics: {metrics_path}")
    print(f"Da luu classification report: {report_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN model for 3-class sentiment analysis."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--num-words", type=int, default=DEFAULT_NUM_WORDS)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dense-units", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size for a quick smoke test.",
    )
    parser.add_argument(
        "--no-nltk-download",
        action="store_true",
        help="Khong tu dong download NLTK resources neu local resource bi thieu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.random_state)

    df = pd.read_csv(args.input)
    if args.sample_size is not None:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=args.random_state)
    preprocessing = prepare_cnn_lstm_data(
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

    vocab_size = min(args.num_words, len(preprocessing.tokenizer.word_index) + 1)
    print(f"Vocabulary size used by embedding: {vocab_size:,}")
    print(f"Train shape: {preprocessing.X_train.shape}")
    print(f"Val shape:   {preprocessing.X_val.shape}")
    print(f"Test shape:  {preprocessing.X_test.shape}")
    print(f"Class weights: {preprocessing.class_weights}")

    model = build_model(
        vocab_size=vocab_size,
        max_len=args.max_len,
        num_classes=len(LABEL_ORDER),
        embedding_dim=args.embedding_dim,
        filters=args.filters,
        kernel_size=args.kernel_size,
        dense_units=args.dense_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )
    model.summary()

    history = train_model(
        model=model,
        X_train=preprocessing.X_train,
        y_train=preprocessing.y_train,
        X_val=preprocessing.X_val,
        y_val=preprocessing.y_val,
        class_weights=preprocessing.class_weights,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    plot_training_history(history, figures_dir=args.figures_dir)
    evaluate_model(
        model=model,
        X_test=preprocessing.X_test,
        y_test=preprocessing.y_test,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
