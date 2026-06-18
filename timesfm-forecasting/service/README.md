# Velents Forecast Service

A stateless HTTP wrapper around **TimesFM 2.5** for the Velents global tenant
assistant (BRD Phase 3 / §10).

## What it does

- `POST /forecast` — takes a numeric series + config, returns a point forecast
  and (optionally) a calibrated q10/q50/q90 uncertainty band.
- `GET /health` — liveness; reports whether the model is loaded.

It is deliberately **tenant-agnostic**: it only ever receives an array of
numbers. The velentsAgents backend pulls the series under the tenant's DB
connection and re-attaches future timestamps to the response, so no tenant
data or identifier ever reaches this service.

## Request / response

```jsonc
// POST /forecast
{
  "values": [12, 9, 14, 18, ...],   // observed, oldest → newest
  "horizon": 14,                     // steps to forecast
  "include_quantiles": true,         // q10/q50/q90 band
  "positive": true,                  // counts/volumes clamp ≥ 0 (infer_is_positive)
  "mode": "forecast"
}

// 200 OK
{
  "status": "ok",                    // or "insufficient_history"
  "horizon": 14,
  "point": [ ... ],                  // median point forecast
  "q10":   [ ... ],
  "q50":   [ ... ],
  "q90":   [ ... ]
}
```

If fewer than `TIMESFM_MIN_HISTORY` (default 16) points are supplied, the
service returns `{"status":"insufficient_history", ...}` rather than a
misleading line (BRD TF-3 / FR-16 / AC-8).

## Config (env)

| Var | Default | Notes |
|-----|---------|-------|
| `TIMESFM_MODEL_ID` | `google/timesfm-2.5-200m-pytorch` | HF checkpoint |
| `TIMESFM_MAX_CONTEXT` | `2048` | context truncation |
| `TIMESFM_MAX_HORIZON` | `256` | max steps |
| `TIMESFM_MIN_HISTORY` | `16` | min observed points to forecast |
| `HF_HOME` | `/models` | weight cache (mount a PVC in k8s) |

## Run

```bash
# from the timesfm repo root
pip install ".[torch]" && pip install -r timesfm-forecasting/service/requirements.txt
uvicorn timesfm-forecasting.service.main:app --host 0.0.0.0 --port 8124
# or via Docker (build context = repo root):
docker build -f timesfm-forecasting/service/Dockerfile -t velents-forecast .
```

Built + deployed by the Jenkins pipeline `devops:aws/ml/prod/jenkins-pipelines/timesfm`
(pushes to ECR `…/timesfm`, deploy chart `devops:aws/ml/prod/helm-charts/timesfm`).
Port **8124** matches that chart's containerPort + `/health` probes. ~2 GB VRAM
(or CPU) for the 200M checkpoint.
