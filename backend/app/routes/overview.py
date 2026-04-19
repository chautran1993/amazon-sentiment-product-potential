"""Overview dashboard route."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas.overview import OverviewResponse
from backend.app.services.product_service import ProductService


router = APIRouter(tags=["overview"])
product_service = ProductService()


@router.get("/overview", response_model=OverviewResponse)
def get_overview() -> OverviewResponse:
    """Return review/product counts and sentiment distribution."""
    return product_service.get_overview()
