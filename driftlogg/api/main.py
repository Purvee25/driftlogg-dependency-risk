"""FastAPI application exposing the risk scorer.

Run locally:
    uvicorn driftlogg.api.main:app --reload

Then open http://localhost:8000 for the dashboard, or /docs for the API.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from driftlogg.api.manifests import ManifestParseError, parse_manifest
from driftlogg.api.schemas import (
    HealthResponse,
    PackageRisk,
    ScoreRequest,
    ScoreResponse,
)
from driftlogg.api.service import HORIZON_DAYS, ScoringService
from driftlogg.config import settings

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 1_000_000
MAX_PACKAGES_PER_REQUEST = 200
"""Each package costs several API calls and seconds of latency; cap the batch."""

_service: ScoringService | None = None


def get_service() -> ScoringService:
    """Provide the shared scoring service.

    Loading the model is expensive, so one instance is reused across requests.
    """
    if _service is None:
        raise HTTPException(status_code=503, detail="Service still starting up.")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup rather than per request."""
    global _service
    logging.basicConfig(level=logging.INFO)
    _service = ScoringService()
    logger.info("Scoring service ready (%s).", _service.model_kind)
    yield
    _service = None


app = FastAPI(
    title="DriftLogg",
    description="Predicts which dependencies are heading for abandonment.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(service: ScoringService = Depends(get_service)) -> HealthResponse:
    """Service status and which model is answering."""
    return HealthResponse(
        status="ok",
        model_loaded=service.model_loaded,
        model_kind=service.model_kind,
        github_token_present=bool(settings.github_token),
    )


def _build_response(results: list[PackageRisk], service: ScoringService) -> ScoreResponse:
    """Wrap scored packages in a response, highest risk first."""
    ordered = sorted(
        results,
        key=lambda r: (r.score if r.score is not None else -1),
        reverse=True,
    )
    return ScoreResponse(
        generated_at=datetime.utcnow(),
        model_kind=service.model_kind,
        horizon_days=HORIZON_DAYS,
        total=len(results),
        scored=sum(1 for r in results if r.scored),
        results=ordered,
    )


@app.post("/score", response_model=ScoreResponse)
def score_packages(
    request: ScoreRequest,
    service: ScoringService = Depends(get_service),
) -> ScoreResponse:
    """Score an explicit list of package names."""
    return _build_response(service.score_packages(request.packages), service)


@app.post("/score/manifest", response_model=ScoreResponse)
async def score_manifest(
    file: UploadFile = File(...),
    include_dev: bool = True,
    service: ScoringService = Depends(get_service),
) -> ScoreResponse:
    """Score every dependency in an uploaded package.json or requirements.txt.

    Raises:
        HTTPException: 400 if the manifest is unparseable or too large.
    """
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Manifest too large.")

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Manifest must be UTF-8 text.") from exc

    try:
        _, names = parse_manifest(file.filename or "", content, include_dev=include_dev)
    except ManifestParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not names:
        raise HTTPException(status_code=400, detail="No dependencies found in manifest.")

    if len(names) > MAX_PACKAGES_PER_REQUEST:
        logger.info("Truncating %d dependencies to %d.", len(names), MAX_PACKAGES_PER_REQUEST)
        names = names[:MAX_PACKAGES_PER_REQUEST]

    return _build_response(service.score_packages(names), service)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
