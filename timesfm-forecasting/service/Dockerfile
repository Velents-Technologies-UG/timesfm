# TimesFM 2.5 forecast service.
#
# IMPORTANT: build context is the REPOSITORY ROOT, not this directory:
#     docker build -f timesfm-forecasting/service/Dockerfile -t timesfm .
#
# Building from the root lets us install the in-repo `timesfm` package (this
# repo's code is the source of truth) instead of the PyPI release, which lags
# the 2.5 code here.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TIMESFM_MODEL_ID=google/timesfm-2.5-200m-pytorch

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch. A plain `pip install torch` pulls the multi-GB CUDA wheel that
# is dead weight on CPU spot nodes, so install from the CPU wheel index instead.
RUN pip install "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cpu

# Web-server deps. torch is installed above; the timesfm package is installed
# from source below (it brings numpy/huggingface_hub/safetensors via pyproject).
RUN pip install \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.29.0" \
        "numpy>=1.26.0" \
        "pydantic>=2.7.0"

# Install the TimesFM package from THIS repo (not PyPI).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Bake the model weights (~800 MB) into the image so pods become ready without a
# HuggingFace download on every cold start / autoscale event.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${TIMESFM_MODEL_ID}')"

# Service code.
COPY timesfm-forecasting/service/main.py ./main.py

# Run as non-root.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/hf-cache
USER appuser

EXPOSE 8124

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8124/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8124"]
