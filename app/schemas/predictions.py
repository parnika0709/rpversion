"""Prediction schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict

from braintumor.schemas.contracts import TumorFeaturesBundle


class PredictionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    case_id: int
    model_name: str
    model_version: str
    inference_seconds: float
    mask_path: str
    uncertainty_path: str | None = None
    saliency_path: str | None = None
    mesh_paths: dict[str, Any]
    features: TumorFeaturesBundle | dict[str, Any]
    metrics: dict[str, Any] | None = None
    risk_regions: list[dict[str, Any]] | None = None
    risk_report: str | None = None
    created_at: datetime
