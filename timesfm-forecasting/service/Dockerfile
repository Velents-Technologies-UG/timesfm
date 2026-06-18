# Forecast microservice image for the Velents global assistant (BRD Phase 3).
# Build context is the timesfm repo ROOT so we can install the vendored package:
#   docker build -f timesfm-forecasting/service/Dockerfile -t velents-forecast .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# Install the vendored TimesFM package (PyTorch backend) + the API server deps.
COPY pyproject.toml README.md ./
COPY src ./src
COPY timesfm-forecasting/service/requirements.txt ./api-requirements.txt
RUN pip install ".[torch]" && pip install -r api-requirements.txt

COPY timesfm-forecasting/service/main.py ./main.py

# Model weights download to HF_HOME on first request; mount a PVC there in k8s
# to avoid re-downloading on every pod restart. Port 8124 matches the deploy
# chart's containerPort + /health probes (aws/ml/prod/helm-charts/timesfm).
EXPOSE 8124
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8124"]
