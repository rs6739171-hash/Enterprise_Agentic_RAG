"""Evaluation routes on the same loopback API used by the protected UI."""
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from evals.hosted import REPORT_DIR, evaluation_status, start_evaluation

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvaluationRequest(BaseModel):
    limit: int = Field(default=3, ge=1, le=15)
    compare_baseline: bool = False


@router.post("/start", status_code=202)
def start(request: EvaluationRequest):
    if not start_evaluation(request.limit, request.compare_baseline):
        raise HTTPException(409, "An evaluation is already running.")
    return evaluation_status()


@router.get("/status")
def status():
    return evaluation_status()


@router.get("/report")
def report():
    path = REPORT_DIR / "latest.json"
    if not path.exists():
        raise HTTPException(404, "No report yet. Start an evaluation.")
    return json.loads(path.read_text(encoding="utf-8"))
