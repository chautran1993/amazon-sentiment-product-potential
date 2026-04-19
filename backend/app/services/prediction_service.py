"""Review sentiment prediction service."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from backend.app.core.config import get_settings
from backend.app.schemas.prediction import ReviewPredictionResponse


LABELS = ["Negative", "Neutral", "Positive"]
POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "best",
    "excellent",
    "good",
    "great",
    "love",
    "perfect",
    "recommend",
    "works",
}
NEGATIVE_WORDS = {
    "bad",
    "broken",
    "defective",
    "disappointed",
    "hate",
    "poor",
    "return",
    "terrible",
    "useless",
    "worst",
}


class PredictionService:
    """Predict review sentiment using a saved model or lightweight fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def predict(self, review_text: str) -> ReviewPredictionResponse:
        """Predict sentiment label and probabilities if available."""
        model = self.load_model(self.settings.baseline_model_path)
        if model is not None:
            return self.predict_with_model(model, review_text)

        return self.predict_with_heuristic(review_text)

    @staticmethod
    @lru_cache(maxsize=1)
    def load_model(path: Path):
        """Load optional saved sklearn pipeline."""
        if not path.exists():
            return None
        return joblib.load(path)

    @staticmethod
    def predict_with_model(model, review_text: str) -> ReviewPredictionResponse:
        """Predict with a saved sklearn-like model."""
        predicted_label = str(model.predict([review_text])[0])
        probabilities = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([review_text])[0]
            classes = [str(label) for label in model.classes_]
            probabilities = {
                label: float(prob)
                for label, prob in zip(classes, proba)
            }

        return ReviewPredictionResponse(
            predicted_label=predicted_label,
            probabilities=probabilities,
            model_name="tfidf_logreg",
        )

    @staticmethod
    def predict_with_heuristic(review_text: str) -> ReviewPredictionResponse:
        """Fallback prediction for demos when no serialized ML model exists."""
        words = set(re.findall(r"[a-z]+", review_text.lower()))
        positive_hits = len(words & POSITIVE_WORDS)
        negative_hits = len(words & NEGATIVE_WORDS)

        negative_score = 1.0 + negative_hits
        neutral_score = 1.0
        positive_score = 1.0 + positive_hits

        scores = np.array([negative_score, neutral_score, positive_score], dtype=float)
        probabilities_array = scores / scores.sum()
        probabilities = {
            label: float(prob)
            for label, prob in zip(LABELS, probabilities_array)
        }
        predicted_label = max(probabilities, key=probabilities.get)

        return ReviewPredictionResponse(
            predicted_label=predicted_label,
            probabilities=probabilities,
            model_name="keyword_heuristic_fallback",
        )

