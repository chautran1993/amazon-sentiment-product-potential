"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas.health import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(status="ok", service="amazon-sentiment-api")

