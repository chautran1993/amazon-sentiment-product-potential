"""Application settings for the backend API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurable paths used by services."""

    project_root: Path = Path(__file__).resolve().parents[3]
    reviews_csv_path: Path = project_root / "data" / "processed" / "labeled_reviews.csv"
    ranking_csv_path: Path = project_root / "outputs" / "ranking" / "product_ranking.csv"
    baseline_model_path: Path = project_root / "models" / "baseline" / "tfidf_logreg_model.joblib"

    model_config = SettingsConfigDict(env_prefix="AMAZON_SENTIMENT_")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()

