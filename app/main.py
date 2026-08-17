"""FastAPI inference service for the Cats vs Dogs classifier.

Endpoints:
    GET  /health   — liveness + model-loaded flag
    POST /predict  — multipart image upload -> label + class probabilities
    GET  /metrics  — Prometheus-format request count & latency metrics

Every request is logged with method, path, status, and latency. Request
bodies (image bytes) are never logged.
"""

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.models.model import load_model
from src.utils import get_eval_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("inference")

MODEL_PATH = Path("models/model.pt")

REQUEST_COUNT = Counter(
    "inference_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds", "Request latency in seconds", ["path"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Predictions served, by predicted label", ["label"]
)

state: dict = {"model": None, "class_names": [], "transform": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, class_names, img_size = load_model(str(MODEL_PATH))
    state.update(
        model=model, class_names=class_names, transform=get_eval_transforms(img_size)
    )
    logger.info("model loaded from %s (classes=%s)", MODEL_PATH, class_names)
    yield
    state["model"] = None


app = FastAPI(title="Cats vs Dogs Inference API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    path = request.url.path
    REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
    REQUEST_LATENCY.labels(path).observe(latency)
    logger.info(
        "%s %s status=%d latency_ms=%.1f", request.method, path, response.status_code, latency * 1000
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file")

    tensor = state["transform"](image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(state["model"](tensor), dim=1)[0]

    probs = {name: round(float(p), 4) for name, p in zip(state["class_names"], probabilities)}
    label = max(probs, key=probs.get)
    PREDICTION_COUNT.labels(label).inc()
    return {"label": label, "probabilities": probs}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
