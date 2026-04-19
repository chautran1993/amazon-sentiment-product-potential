"""Prediction request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReviewPredictionRequest(BaseModel):
    review_text: str = Field(..., min_length=1, examples=["This product works great."])


class ReviewPredictionResponse(BaseModel):
    predicted_label: str
    probabilities: dict[str, float] | None = None
    model_name: str

