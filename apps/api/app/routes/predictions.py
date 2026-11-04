"""Prediction endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException
from app.config import settings
from app.services import CaseService, InferenceService
from app.schemas import PredictionRead

router = APIRouter(prefix="/cases", tags=["Predictions"])
case_service = CaseService()
inference_service = InferenceService()


@router.post("/{case_id}/predict", status_code=202)
async def run_prediction(case_id: int):
    """Trigger ML inference for a case."""
    case = case_service.get_case_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")

    # The volume is stored in the upload_dir specified in Case.nifti_path
    case_dir = Path(case.nifti_path)

    try:
        prediction = inference_service.run_inference(case_id, case_dir)
        return {"message": "Inference completed successfully", "prediction_id": prediction.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/{case_id}/prediction", response_model=PredictionRead | None)
async def get_latest_prediction(case_id: int):
    """Get the most recent prediction for a case."""
    prediction = case_service.get_latest_case_prediction(case_id)
    return prediction
