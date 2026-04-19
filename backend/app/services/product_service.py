"""Services for loading products and ranking data from CSV files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.app.core.config import get_settings
from backend.app.schemas.overview import OverviewResponse, SentimentCount
from backend.app.schemas.products import ProductDetail, ProductRankingItem


SENTIMENT_SCORE_MAP = {
    "Negative": 1,
    "Neutral": 2,
    "Positive": 3,
}


class ProductService:
    """Read products from ranking CSV or aggregate from reviews CSV."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def list_products(self, limit: int = 20, offset: int = 0) -> tuple[list[ProductRankingItem], int]:
        """Return paginated product ranking items."""
        ranking = self.load_or_build_ranking()
        if ranking.empty:
            return [], 0

        total = len(ranking)
        page = ranking.iloc[offset : offset + limit]
        return [self.row_to_ranking_item(row) for _, row in page.iterrows()], total

    def get_top_ranking(self, limit: int = 20) -> list[ProductRankingItem]:
        """Return top ranked products."""
        ranking = self.load_or_build_ranking()
        if ranking.empty:
            return []
        return [self.row_to_ranking_item(row) for _, row in ranking.head(limit).iterrows()]

    def get_product_detail(self, asin: str) -> ProductDetail | None:
        """Return one product with optional sample reviews."""
        ranking = self.load_or_build_ranking()
        if ranking.empty:
            return None

        match = ranking[ranking["asin"].astype(str) == asin]
        if match.empty:
            return None

        item = self.row_to_ranking_item(match.iloc[0])
        sample_reviews = self.get_sample_reviews(asin)
        sentiment_distribution = self.get_product_sentiment_distribution(asin)
        return ProductDetail(
            **item.model_dump(),
            sample_reviews=sample_reviews,
            sentiment_distribution=sentiment_distribution,
        )

    def get_overview(self) -> OverviewResponse:
        """Return dashboard overview stats from reviews and ranking CSV files."""
        reviews = self.load_reviews_csv(self.settings.reviews_csv_path)
        ranking = self.load_or_build_ranking()

        total_reviews = int(len(reviews)) if not reviews.empty else 0
        if not ranking.empty:
            total_products = int(len(ranking))
        elif not reviews.empty and "asin" in reviews.columns:
            total_products = int(reviews["asin"].nunique())
        else:
            total_products = 0

        sentiment_distribution = []
        sentiment_col = self.get_sentiment_column(reviews)
        if sentiment_col:
            counts = reviews[sentiment_col].dropna().astype(str).value_counts()
            for label in ["Negative", "Neutral", "Positive"]:
                sentiment_distribution.append(
                    SentimentCount(label=label, count=int(counts.get(label, 0)))
                )

        return OverviewResponse(
            total_reviews=total_reviews,
            total_products=total_products,
            sentiment_distribution=sentiment_distribution,
        )

    def load_or_build_ranking(self) -> pd.DataFrame:
        """Load product ranking CSV, or build a simple ranking from reviews CSV."""
        ranking = self.load_ranking_csv(self.settings.ranking_csv_path)
        if not ranking.empty:
            return ranking

        return self.build_ranking_from_reviews(self.settings.reviews_csv_path)

    @staticmethod
    @lru_cache(maxsize=4)
    def load_reviews_csv(path: Path) -> pd.DataFrame:
        """Load review CSV if available."""
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    @staticmethod
    @lru_cache(maxsize=4)
    def load_ranking_csv(path: Path) -> pd.DataFrame:
        """Load precomputed product ranking CSV if available."""
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)
        if "asin" not in df.columns:
            return pd.DataFrame()

        if "product_potential_score" in df.columns:
            df = df.sort_values("product_potential_score", ascending=False)
        return df.reset_index(drop=True)

    @staticmethod
    @lru_cache(maxsize=4)
    def build_ranking_from_reviews(path: Path) -> pd.DataFrame:
        """Build product ranking from labeled review CSV when asin is present."""
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)
        sentiment_col = "predicted_sentiment" if "predicted_sentiment" in df.columns else "sentiment_label"
        if "asin" not in df.columns or sentiment_col not in df.columns:
            return pd.DataFrame()

        working = df[["asin", sentiment_col]].dropna().copy()
        working["asin"] = working["asin"].astype(str).str.strip()
        working = working[working["asin"] != ""]
        working["sentiment_score"] = working[sentiment_col].map(SENTIMENT_SCORE_MAP)
        working = working.dropna(subset=["sentiment_score"])

        grouped = working.groupby("asin")
        ranking = grouped.agg(
            review_count=("sentiment_score", "size"),
            sentiment_score_mean=("sentiment_score", "mean"),
            positive_count=("sentiment_score", lambda values: (values == 3).sum()),
            negative_count=("sentiment_score", lambda values: (values == 1).sum()),
        ).reset_index()

        ranking["positive_ratio"] = ranking["positive_count"] / ranking["review_count"]
        ranking["negative_ratio"] = ranking["negative_count"] / ranking["review_count"]
        ranking["normalized_sentiment_mean"] = ProductService.min_max_normalize(
            ranking["sentiment_score_mean"]
        )
        ranking["normalized_review_volume"] = ProductService.min_max_normalize(
            ranking["review_count"]
        )
        ranking["product_potential_score"] = (
            0.7 * ranking["normalized_sentiment_mean"]
            + 0.3 * ranking["normalized_review_volume"]
        )
        return ranking.sort_values("product_potential_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def min_max_normalize(series: pd.Series) -> pd.Series:
        """Normalize a pandas series to 0-1 range."""
        min_value = series.min()
        max_value = series.max()
        if max_value == min_value:
            return pd.Series(0.0, index=series.index)
        return (series - min_value) / (max_value - min_value)

    def get_sample_reviews(self, asin: str, limit: int = 5) -> list[str]:
        """Return sample review texts for a product when review CSV contains asin."""
        path = self.settings.reviews_csv_path
        if not path.exists():
            return []

        df = self.load_reviews_csv(path)
        if "asin" not in df.columns or "reviewText" not in df.columns:
            return []

        reviews = df[df["asin"].astype(str) == asin]["reviewText"].dropna().astype(str).head(limit)
        return reviews.tolist()

    def get_product_sentiment_distribution(self, asin: str) -> list[dict[str, int]]:
        """Return sentiment distribution for one product."""
        df = self.load_reviews_csv(self.settings.reviews_csv_path)
        sentiment_col = self.get_sentiment_column(df)
        if df.empty or "asin" not in df.columns or sentiment_col is None:
            return []

        product_reviews = df[df["asin"].astype(str) == asin]
        counts = product_reviews[sentiment_col].dropna().astype(str).value_counts()
        return [
            {"label": label, "count": int(counts.get(label, 0))}
            for label in ["Negative", "Neutral", "Positive"]
        ]

    @staticmethod
    def get_sentiment_column(df: pd.DataFrame) -> str | None:
        """Return the available sentiment column name."""
        if df.empty:
            return None
        if "predicted_sentiment" in df.columns:
            return "predicted_sentiment"
        if "sentiment_label" in df.columns:
            return "sentiment_label"
        return None

    @staticmethod
    def row_to_ranking_item(row: pd.Series) -> ProductRankingItem:
        """Convert a dataframe row to API schema."""
        return ProductRankingItem(
            asin=str(row.get("asin", "")),
            review_count=int(row.get("review_count", 0) or 0),
            sentiment_score_mean=ProductService.safe_float(row.get("sentiment_score_mean")),
            positive_ratio=ProductService.safe_float(row.get("positive_ratio")),
            negative_ratio=ProductService.safe_float(row.get("negative_ratio")),
            product_potential_score=ProductService.safe_float(row.get("product_potential_score")),
        )

    @staticmethod
    def safe_float(value: object) -> float | None:
        """Convert numeric values to float, preserving missing values as None."""
        if pd.isna(value):
            return None
        return float(value)
