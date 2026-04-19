"""Train a BiLSTM sentiment classifier with TensorFlow/Keras.

The model uses a Bidirectional LSTM because sentiment in reviews can depend on
context from both earlier and later words. The default hyperparameters are kept
moderate so the script is practical on a student laptop CPU.
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
    precision_score,
    recall_score,
)
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam

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
DEFAULT_MODEL_DIR = "models/lstm"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_OUTPUT_DIR = "outputs/lstm"
RANDOM_STATE = 42


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    """Make training more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(
    vocab_size: int,
    max_len: int = DEFAULT_MAX_LEN,
    num_classes: int = 3,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    dense_units: int = 64,
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-3,
    bidirectional: bool = True,
) -> tf.keras.Model:
    """Build and compile an LSTM/BiLSTM model for 3-class sentiment analysis."""
    layers = [
        Input(shape=(max_len,), dtype="int32", name="input_tokens"),
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="embedding",
        ),
    ]

    lstm_layer = LSTM(lstm_units, dropout=0.2, name="lstm")
    if bidirectional:
        layers.append(Bidirectional(lstm_layer, name="bidirectional_lstm"))
    else:
        layers.append(lstm_layer)

    layers.extend(
        [
            Dense(dense_units, activation="relu", name="dense_hidden"),
            Dropout(dropout_rate, name="dropout"),
            Dense(num_classes, activation="softmax", name="sentiment_output"),
        ]
    )

    model = Sequential(layers, name="lstm_sentiment_classifier")
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
    early_stop_patience: int = 3,
    lr_patience: int = 2,
) -> tf.keras.callbacks.History:
    """Train model with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best_model.weights.h5"
    model_config_path = model_dir / "model_config.json"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-6,
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


def plot_training_history(
    history: tf.keras.callbacks.History,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Save separate train/validation loss and accuracy charts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title("LSTM Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = output_dir / "lstm_training_loss.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Da luu loss plot: {loss_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title("LSTM Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    accuracy_path = output_dir / "lstm_training_accuracy.png"
    plt.savefig(accuracy_path, dpi=150)
    plt.close()
    print(f"Da luu accuracy plot: {accuracy_path}")


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Save confusion matrix chart."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lstm_confusion_matrix.png"

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_ORDER))))

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("LSTM Confusion Matrix")
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
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, float]:
    """Evaluate model and save metrics/report/confusion matrix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    matrix = confusion_matrix(y_test, y_pred, labels=list(range(len(LABEL_ORDER))))
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(LABEL_ORDER))),
        target_names=LABEL_ORDER,
        zero_division=0,
    )

    print("\n=== LSTM Test Metrics ===")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision macro: {metrics['precision_macro']:.4f}")
    print(f"Recall macro:    {metrics['recall_macro']:.4f}")
    print(f"Macro F1-score:  {metrics['macro_f1']:.4f}")

    print("\n=== Confusion Matrix ===")
    print(pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER).to_string())

    print("\n=== Classification Report ===")
    print(report)

    save_confusion_matrix_plot(y_test, y_pred, output_dir=output_dir)

    metrics_path = output_dir / "lstm_metrics.json"
    report_path = output_dir / "lstm_classification_report.txt"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with report_path.open("w", encoding="utf-8") as file:
        file.write(report)

    print(f"Da luu metrics: {metrics_path}")
    print(f"Da luu classification report: {report_path}")
    return metrics


def load_best_model(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Load model architecture and best weights from models/lstm/."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM/BiLSTM model for Amazon review sentiment analysis."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--num-words", type=int, default=DEFAULT_NUM_WORDS)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Dung LSTM mot chieu thay vi BiLSTM.",
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
    print("Using BiLSTM by default: it captures left and right context in reviews.")
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
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        bidirectional=not args.use_lstm,
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
        early_stop_patience=args.early_stop_patience,
        lr_patience=args.lr_patience,
    )

    plot_training_history(history, output_dir=args.output_dir)
    evaluate_model(
        model=model,
        X_test=preprocessing.X_test,
        y_test=preprocessing.y_test,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
