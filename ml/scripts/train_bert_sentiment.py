"""Fine-tune bert-base-uncased for 3-class English review sentiment.

Designed for student projects and Google Colab GPU:
- Uses Hugging Face tokenizer/model.
- Keeps max_len and epochs moderate by default.
- Supports --sample-size for quick smoke tests.
- Saves best model and tokenizer to models/bert/.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler


DEFAULT_INPUT_PATH = "data/processed/labeled_reviews.csv"
DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_MODEL_DIR = "models/bert"
DEFAULT_OUTPUT_DIR = "outputs/bert"
RANDOM_STATE = 42
LABEL_ORDER = ["Negative", "Neutral", "Positive"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


@dataclass
class SplitData:
    """Container for train/validation/test text and label splits."""

    X_train: pd.Series
    X_val: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


class ReviewDataset(Dataset):
    """Torch dataset that tokenizes review text for BERT."""

    def __init__(
        self,
        texts: pd.Series | list[str],
        labels: pd.Series | list[int],
        tokenizer: AutoTokenizer,
        max_len: int,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    """Make training more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use GPU when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_labeled_reviews(input_path: str | Path) -> pd.DataFrame:
    """Load reviewText and sentiment_label from labeled_reviews.csv."""
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
    df["label_id"] = df["sentiment_label"].map(LABEL2ID)

    if df.empty:
        raise ValueError("Dataset rong sau khi loc reviewText va sentiment_label.")

    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> SplitData:
    """Split data into stratified train/validation/test sets."""
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size va val_size phai > 0 va tong nho hon 1.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        df["reviewText"],
        df["label_id"],
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=df["label_id"],
    )

    relative_test_size = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp,
    )

    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def create_dataloaders(
    splits: SplitData,
    tokenizer: AutoTokenizer,
    max_len: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test dataloaders."""
    train_dataset = ReviewDataset(splits.X_train, splits.y_train, tokenizer, max_len)
    val_dataset = ReviewDataset(splits.X_val, splits.y_val, tokenizer, max_len)
    test_dataset = ReviewDataset(splits.X_test, splits.y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_model(model_name: str = DEFAULT_MODEL_NAME) -> AutoModelForSequenceClassification:
    """Load BERT for sequence classification with 3 sentiment labels."""
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


def compute_class_weights(labels: pd.Series, device: torch.device) -> torch.Tensor:
    """Compute balanced class weights for CrossEntropyLoss."""
    classes = np.array(list(range(len(LABEL_ORDER))))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels.to_numpy(),
    )
    return torch.tensor(weights, dtype=torch.float, device=device)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(dataloader), 1)


def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run prediction and return true labels, predicted labels, and average loss."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            preds = torch.argmax(outputs.logits, dim=1)

            total_loss += outputs.loss.item()
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    return np.array(y_true), np.array(y_pred), avg_loss


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: pd.Series,
    tokenizer: AutoTokenizer,
    model_dir: str | Path,
    device: torch.device,
    epochs: int = 2,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    use_class_weights: bool = True,
) -> dict[str, list[float]]:
    """Fine-tune BERT and save the best validation macro-F1 checkpoint."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    if use_class_weights:
        class_weights = compute_class_weights(y_train, device=device)
        print(f"Class weights: {class_weights.detach().cpu().numpy().round(4).tolist()}")
    else:
        class_weights = None

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }
    best_macro_f1 = -1.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
        )
        y_val_true, y_val_pred, val_loss = predict(model, val_loader, device=device)
        val_accuracy = accuracy_score(y_val_true, y_val_pred)
        val_macro_f1 = f1_score(y_val_true, y_val_pred, average="macro", zero_division=0)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_accuracy))
        history["val_macro_f1"].append(float(val_macro_f1))

        print(f"Train loss:   {train_loss:.4f}")
        print(f"Val loss:     {val_loss:.4f}")
        print(f"Val accuracy: {val_accuracy:.4f}")
        print(f"Val macro F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"Saved best model/tokenizer to {model_dir}")

    with (model_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    return history


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
) -> Path:
    """Save confusion matrix as a matplotlib image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bert_confusion_matrix.png"

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_ORDER))))

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("BERT Confusion Matrix")
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
    model: torch.nn.Module,
    test_loader: DataLoader,
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate BERT on test set and save metrics/report/confusion matrix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, test_loss = predict(model, test_loader, device=device)
    metrics = {
        "test_loss": float(test_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_ORDER))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABEL_ORDER))),
        target_names=LABEL_ORDER,
        zero_division=0,
    )

    print("\n=== BERT Test Metrics ===")
    print(f"Test loss:      {metrics['test_loss']:.4f}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision_macro']:.4f}")
    print(f"Recall:         {metrics['recall_macro']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")

    print("\n=== Confusion Matrix ===")
    print(pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER).to_string())

    print("\n=== Classification Report ===")
    print(report)

    save_confusion_matrix_plot(y_true, y_pred, output_dir=output_dir)

    with (output_dir / "bert_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with (output_dir / "bert_classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(report)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune bert-base-uncased for 3-class sentiment analysis."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Tat class weights neu muon loss khong can bang lop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.random_state)

    device = get_device()
    print(f"Using device: {device}")

    df = load_labeled_reviews(args.input)
    if args.sample_size is not None:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=args.random_state)

    splits = split_data(
        df=df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader, test_loader = create_dataloaders(
        splits=splits,
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )

    print(f"Train samples: {len(splits.X_train):,}")
    print(f"Val samples:   {len(splits.X_val):,}")
    print(f"Test samples:  {len(splits.X_test):,}")
    print(f"Labels: {LABEL2ID}")

    model = build_model(args.model_name)
    training_start = time.perf_counter()
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_train=splits.y_train,
        tokenizer=tokenizer,
        model_dir=args.model_dir,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_class_weights=not args.no_class_weights,
    )
    training_time_seconds = time.perf_counter() - training_start

    best_model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    best_model.to(device)
    metrics = evaluate_model(
        model=best_model,
        test_loader=test_loader,
        output_dir=args.output_dir,
        device=device,
    )
    metrics["training_time_seconds"] = float(training_time_seconds)
    metrics_path = Path(args.output_dir) / "bert_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"Training time:  {training_time_seconds:.2f} seconds")
    print(f"Updated metrics with training time: {metrics_path}")


if __name__ == "__main__":
    main()
