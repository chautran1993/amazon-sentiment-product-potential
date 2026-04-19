"""FastAPI entrypoint for Amazon sentiment analysis demo."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes.health import router as health_router
from backend.app.routes.overview import router as overview_router
from backend.app.routes.predict import router as predict_router
from backend.app.routes.products import router as products_router


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Amazon Sentiment Analysis API",
        description="Demo API for sentiment prediction and product potential ranking.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(overview_router)
    app.include_router(products_router)
    app.include_router(predict_router)
    return app


app = create_app()
