"""
TimesFM forecast microservice (Velents global assistant — BRD Phase 3 / §10).

A thin, STATELESS HTTP wrapper around TimesFM 2.5. It receives a numeric
series (already pulled and tenant-scoped by the velentsAgents backend) plus a
config, runs the model, and returns a structure that maps directly onto the
assistant's forecast/fan chart: {point, q10, q50, q90}.

Design rules from the BRD:
  - The service NEVER sees a tenant identifier or raw tenant rows — only an
    array of numbers. Tenant isolation lives entirely in the PHP layer.
  - The LLM never runs the model; it calls this service through one tool.
  - A minimum-history guard returns a structured `insufficient_history`
    instead of a confident-looking wrong line (FR-16 / TF-3).

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8200
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Literal

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger("forecast")
logging.basicConfig(level=logging.INFO)

# Tunables (env-overridable so the Helm chart can size them per node).
MODEL_ID = os.getenv("TIMESFM_MODEL_ID", "google/timesfm-2.5-200m-pytorch")
MAX_CONTEXT = int(os.getenv("TIMESFM_MAX_CONTEXT", "2048"))
MAX_HORIZON = int(os.getenv("TIMESFM_MAX_HORIZON", "256"))
# Minimum observed points before we are willing to forecast at all.
MIN_HISTORY = int(os.getenv("TIMESFM_MIN_HISTORY", "16"))

# Quantile head column order is [mean, q10, q20, ..., q90] (10 columns).
Q10_IDX, Q50_IDX, Q90_IDX = 1, 5, 9

app = FastAPI(title="Velents Forecast Service", version="1.0")

# The model is heavy to load and must be compiled once; guard with a lock so
# the first concurrent requests don't each try to compile.
_model = None
_model_lock = threading.Lock()


def _get_model(infer_is_positive: bool):
    """Lazily load + compile TimesFM 2.5. Compilation is config-bound, so we
    compile once for the worst-case (max) context/horizon and reuse it."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        import timesfm  # imported lazily so /health works without the model

        logger.info("Loading TimesFM checkpoint %s", MODEL_ID)
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
        model.compile(
            timesfm.ForecastConfig(
                max_context=MAX_CONTEXT,
                max_horizon=MAX_HORIZON,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=infer_is_positive,
                fix_quantile_crossing=True,
            )
        )
        _model = model
        logger.info("TimesFM compiled (max_context=%s, max_horizon=%s)", MAX_CONTEXT, MAX_HORIZON)
        return _model


class ForecastRequest(BaseModel):
    # The observed series, oldest → newest. No timestamps: the caller owns the
    # calendar and re-attaches future timestamps to our output.
    values: list[float] = Field(..., min_length=1)
    horizon: int = Field(..., ge=1, le=MAX_HORIZON)
    include_quantiles: bool = True
    # Counts/volumes can't go negative; the caller sets this per metric.
    positive: bool = True
    mode: Literal["forecast", "anomaly"] = "forecast"


class ForecastResponse(BaseModel):
    status: Literal["ok", "insufficient_history"]
    horizon: int = 0
    point: list[float] = []
    q10: list[float] = []
    q50: list[float] = []
    q90: list[float] = []
    reason: str | None = None
    min_history: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": _model is not None}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    series = np.asarray([v for v in req.values if v is not None], dtype=np.float32)

    # Minimum-history guard — refuse rather than fabricate (TF-3 / FR-16 / AC-8).
    if series.size < MIN_HISTORY:
        return ForecastResponse(
            status="insufficient_history",
            reason=f"Need at least {MIN_HISTORY} historical points, got {series.size}.",
            min_history=MIN_HISTORY,
        )

    horizon = min(req.horizon, MAX_HORIZON)
    model = _get_model(infer_is_positive=req.positive)

    point_forecast, quantile_forecast = model.forecast(horizon=horizon, inputs=[series])

    point = np.asarray(point_forecast)[0]
    resp = ForecastResponse(status="ok", horizon=horizon, point=_round(point))

    if req.include_quantiles:
        q = np.asarray(quantile_forecast)[0]  # (horizon, 10)
        resp.q10 = _round(q[:, Q10_IDX])
        resp.q50 = _round(q[:, Q50_IDX])
        resp.q90 = _round(q[:, Q90_IDX])

    return resp


def _round(arr) -> list[float]:
    return [round(float(x), 4) for x in np.asarray(arr).ravel().tolist()]
