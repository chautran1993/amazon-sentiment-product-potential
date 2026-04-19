"""Overview dashboard schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SentimentCount(BaseModel):
    label: str
    count: int


class OverviewResponse(BaseModel):
    total_reviews: int = 0
    total_products: int = 0
    sentiment_distribution: list[SentimentCount] = Field(default_factory=list)

