# TimesFM Forecast Service

A thin FastAPI wrapper around Google's TimesFM 2.5 model. Loads the model once at startup and serves zero-shot univariate forecasts over HTTP.

## Endpoints

### `POST /forecast`

Request:
```json
{
  "data": [12.3, 11.7, 14.1, 13.5, 15.2, 16.8, 18.3],
  "horizon": 14,
  "frequency": "daily"
}
```

Response:
```json
{
  "point_forecast": [19.1, 19.8, 20.4, ...],
  "quantile_low": [17.2, 17.5, 17.9, ...],
  "quantile_high": [21.0, 21.9, 22.6, ...],
  "horizon": 14,
  "input_length": 7,
  "frequency": "daily"
}
```

`quantile_low` is the 10th percentile and `quantile_high` is the 90th percentile — use them as the outer edges of an 80% confidence band.

### `GET /health`

Returns `{"status": "ok", "model_loaded": true, ...}` once the model is ready.

## Run locally

```bash
pip install -r requirements.txt
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8124
```

Model weights (~800 MB) download from HuggingFace on first startup and cache in `~/.cache/huggingface/`.

## Configuration

Environment variables:
- `PORT` — HTTP port (default 8124)
- `TIMESFM_MODEL_ID` — HuggingFace model id (default `google/timesfm-2.5-200m-pytorch`)
- `TIMESFM_MAX_CONTEXT` — maximum input history length (default 1024)
- `TIMESFM_MAX_HORIZON` — maximum forecast horizon (default 256)
- `TIMESFM_BATCH_SIZE` — per-core batch size (default 32)

## Docker

```bash
docker build -t velents/timesfm-service .
docker run -p 8124:8124 velents/timesfm-service
```
