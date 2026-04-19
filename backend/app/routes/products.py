"""Product and ranking routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.products import ProductDetail, ProductListResponse, ProductRankingItem
from backend.app.services.product_service import ProductService


router = APIRouter(prefix="/products", tags=["products"])
product_service = ProductService()


@router.get("", response_model=ProductListResponse)
def list_products(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ProductListResponse:
    """Return products from ranking CSV or review CSV if asin exists."""
    products, total = product_service.list_products(limit=limit, offset=offset)
    return ProductListResponse(total=total, limit=limit, offset=offset, items=products)


@router.get("/top-ranking", response_model=list[ProductRankingItem])
def top_ranking(limit: int = Query(default=20, ge=1, le=100)) -> list[ProductRankingItem]:
    """Return top products by potential score."""
    return product_service.get_top_ranking(limit=limit)


@router.get("/{asin}", response_model=ProductDetail)
def get_product(asin: str) -> ProductDetail:
    """Return aggregated information for one product."""
    product = product_service.get_product_detail(asin)
    if product is None:
        raise HTTPException(status_code=404, detail=f"Product not found: {asin}")
    return product

