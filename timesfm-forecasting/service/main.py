"""FastAPI wrapper around Google's TimesFM 2.5 forecasting model.

Exposes a minimal HTTP surface so other services (e.g. a LangGraph agent)
can request zero-shot time-series forecasts without loading the 200M
parameter model themselves.

Runs on port 8124 by default. Load the model once at startup and reuse
it across requests for sub-second inference.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import timesfm
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("timesfm-service")
logging.basicConfig(level=logging.INFO)

MODEL_ID = os.getenv("TIMESFM_MODEL_ID", "google/timesfm-2.5-200m-pytorch")
MAX_CONTEXT = int(os.getenv("TIMESFM_MAX_CONTEXT", "1024"))
MAX_HORIZON = int(os.getenv("TIMESFM_MAX_HORIZON", "256"))
BATCH_SIZE = int(os.getenv("TIMESFM_BATCH_SIZE", "32"))

_state: dict[str, Any] = {"model": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Loading TimesFM model %s...", MODEL_ID)
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=BATCH_SIZE,
        )
    )
    _state["model"] = model
    logger.info("TimesFM model ready.")
    yield
    _state["model"] = None


app = FastAPI(
    title="TimesFM Forecast Service",
    version="1.0.0",
    description="Zero-shot time-series forecasting via Google's TimesFM.",
    lifespan=lifespan,
)


class ForecastRequest(BaseModel):
    data: list[float] = Field(
        ...,
        min_length=3,
        description="Historical univariate time-series values ordered chronologically.",
    )
    horizon: int = Field(
        default=12,
        ge=1,
        le=MAX_HORIZON,
        description="Number of future periods to forecast.",
    )
    frequency: str = Field(
        default="daily",
        description="Human-readable frequency hint (daily, weekly, monthly, hourly). Not used by the model but echoed in responses.",
    )


class ForecastResponse(BaseModel):
    point_forecast: list[float]
    quantile_low: list[float]
    quantile_high: list[float]
    horizon: int
    input_length: int
    frequency: str


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "model_id": MODEL_ID,
        "max_context": MAX_CONTEXT,
        "max_horizon": MAX_HORIZON,
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if len(req.data) > MAX_CONTEXT:
        # Keep the most recent MAX_CONTEXT points — TimesFM's decoder uses the
        # tail for conditioning and longer inputs get truncated anyway.
        series = req.data[-MAX_CONTEXT:]
    else:
        series = req.data

    try:
        point_forecast, quantile_forecast = model.forecast(
            horizon=req.horizon,
            inputs=[np.array(series, dtype=np.float32)],
        )
    except Exception as exc:
        logger.exception("Forecast failed")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc

    # quantile_forecast shape: [batch_size, horizon, num_quantiles]
    # Quantiles are [0.1, 0.2, ..., 0.9] — index 0 = 10th, index -1 = 90th.
    point = point_forecast[0].tolist()
    q_low = quantile_forecast[0][:, 0].tolist()
    q_high = quantile_forecast[0][:, -1].tolist()

    return ForecastResponse(
        point_forecast=point,
        quantile_low=q_low,
        quantile_high=q_high,
        horizon=req.horizon,
        input_length=len(series),
        frequency=req.frequency,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8124"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
