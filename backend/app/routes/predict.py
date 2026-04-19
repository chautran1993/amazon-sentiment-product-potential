"""Review sentiment prediction route."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas.prediction import ReviewPredictionRequest, ReviewPredictionResponse
from backend.app.services.prediction_service import PredictionService


router = APIRouter(tags=["prediction"])
prediction_service = PredictionService()


@router.post("/predict-review", response_model=ReviewPredictionResponse)
def predict_review(payload: ReviewPredictionRequest) -> ReviewPredictionResponse:
    """Predict sentiment label for a review text."""
    return prediction_service.predict(payload.review_text)

