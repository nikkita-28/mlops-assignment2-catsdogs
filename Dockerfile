# Inference service image — CPU-only PyTorch, model baked in.
FROM python:3.13-slim

WORKDIR /service

# Install pinned dependencies first for layer caching.
COPY requirements-api.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple -r requirements-api.txt

COPY src/ src/
COPY app/ app/
COPY models/model.pt models/model.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
