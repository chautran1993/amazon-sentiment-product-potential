"""Product response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProductRankingItem(BaseModel):
    asin: str
    review_count: int = 0
    sentiment_score_mean: float | None = None
    positive_ratio: float | None = None
    negative_ratio: float | None = None
    product_potential_score: float | None = None


class ProductListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ProductRankingItem]


class ProductDetail(ProductRankingItem):
    sample_reviews: list[str] = Field(default_factory=list)
    sentiment_distribution: list[dict[str, int]] = Field(default_factory=list)
