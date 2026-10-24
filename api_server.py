import json
import logging
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from text_model import encode_review, load_checkpoint, predict_sentiment
from transformer_model import (
    DEFAULT_MODEL_ID,
    load_pretrained_model,
    predict_batch_with_transformer,
    predict_with_transformer,
)


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "movie_review_sentiment_cnn.pth"
METRICS_PATH = ARTIFACTS_DIR / "test_metrics.json"
HISTORY_PATH = ARTIFACTS_DIR / "training_history.json"
STATIC_DIR = PROJECT_DIR / "static"
LOGGER = logging.getLogger("sentiment_studio")
MODEL_LOCK = threading.Lock()
TRAINING_LOCK = threading.Lock()
TRAINING_JOB: dict[str, object] | None = None

app = FastAPI(
    title="Movie Review Sentiment Studio",
    description="Local AI sentiment analysis, model evaluation, and training API.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionRequest(BaseModel):
    review: str = Field(min_length=1, max_length=10_000)
    model: Literal["transformer", "cnn"] = "transformer"


class BatchPredictionRequest(BaseModel):
    reviews: list[str] = Field(min_length=1, max_length=100)
    model: Literal["transformer", "cnn"] = "transformer"


class TrainingRequest(BaseModel):
    epochs: int = Field(default=5, ge=1, le=20)
    batch_size: int = Field(default=64, ge=16, le=256)
    learning_rate: float = Field(default=0.001, gt=0.000001, le=0.01)
    max_train_samples: int | None = Field(default=None, ge=4, le=1800)
    max_test_samples: int | None = Field(default=None, ge=2, le=200)


@lru_cache(maxsize=1)
def get_transformer():
    return load_pretrained_model(DEFAULT_MODEL_ID)


def predict(review: str, model_name: Literal["transformer", "cnn"]) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with MODEL_LOCK:
        try:
            if model_name == "transformer":
                tokenizer, model, device = get_transformer()
                sentiment, confidence, probabilities = predict_with_transformer(
                    review, tokenizer, model, device
                )
                display_name = "DistilBERT pretrained sentiment model"
            else:
                if not CHECKPOINT_PATH.is_file():
                    raise HTTPException(
                        status_code=409,
                        detail="The project CNN is not trained yet. Start training first.",
                    )
                model, vocabulary, max_length = load_checkpoint(CHECKPOINT_PATH, device)
                sentiment, confidence = predict_sentiment(
                    model, review, vocabulary, max_length, device
                )
                probabilities = {}
                display_name = "Project-trained text CNN"
        except HTTPException:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.exception("Sentiment inference failed.")
            raise HTTPException(
                status_code=503,
                detail=(
                    "Could not load or run the selected model. Check your internet "
                    f"connection on its first use. Details: {error}"
                ),
            ) from error

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "scores": probabilities,
        "model": display_name,
        "device": str(device),
        "review_length": len(review),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def predict_batch(
    reviews: list[str], model_name: Literal["transformer", "cnn"]
) -> list[dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with MODEL_LOCK:
        try:
            if model_name == "transformer":
                tokenizer, model, device = get_transformer()
                results = predict_batch_with_transformer(
                    reviews, tokenizer, model, device
                )
                display_name = "DistilBERT pretrained sentiment model"
            else:
                if not CHECKPOINT_PATH.is_file():
                    raise HTTPException(
                        status_code=409,
                        detail="The project CNN is not trained yet. Start training first.",
                    )
                model, vocabulary, max_length = load_checkpoint(CHECKPOINT_PATH, device)
                encoded = [
                    encode_review(review, vocabulary, max_length) for review in reviews
                ]
                lengths = torch.tensor(
                    [len(tokens) for tokens in encoded],
                    dtype=torch.long,
                    device=device,
                )
                token_ids = torch.zeros(
                    (len(encoded), int(lengths.max().item())),
                    dtype=torch.long,
                    device=device,
                )
                for row, tokens in enumerate(encoded):
                    token_ids[row, : len(tokens)] = torch.tensor(
                        tokens, dtype=torch.long, device=device
                    )
                with torch.inference_mode():
                    probabilities = torch.softmax(model(token_ids, lengths), dim=1)
                results = []
                for row in probabilities:
                    index = int(row.argmax().item())
                    sentiment = ("negative", "positive")[index]
                    results.append((sentiment, row[index].item(), {}))
                display_name = "Project-trained text CNN"
        except HTTPException:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.exception("Batch sentiment inference failed.")
            raise HTTPException(
                status_code=503,
                detail=f"Could not run batch inference. Details: {error}",
            ) from error

    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "review": review,
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": scores,
            "model": display_name,
            "device": str(device),
            "review_length": len(review),
            "created_at": now,
        }
        for review, (sentiment, confidence, scores) in zip(reviews, results)
    ]


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(PROJECT_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ready",
        "transformer": DEFAULT_MODEL_ID,
        "cnn_checkpoint_available": CHECKPOINT_PATH.is_file(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/api/metrics")
def metrics() -> dict[str, object]:
    test_metrics = (
        json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if METRICS_PATH.is_file()
        else None
    )
    history = (
        json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if HISTORY_PATH.is_file()
        else None
    )
    return {"test_metrics": test_metrics, "training_history": history}


@app.post("/api/predict")
def single_prediction(request: PredictionRequest) -> dict[str, object]:
    review = request.review.strip()
    if not review:
        raise HTTPException(status_code=422, detail="Enter a non-empty review.")
    return predict(review, request.model)


@app.post("/api/predict/batch")
def batch_prediction(request: BatchPredictionRequest) -> dict[str, object]:
    reviews = [review.strip() for review in request.reviews]
    if any(not review for review in reviews):
        raise HTTPException(status_code=422, detail="Every review must contain text.")
    if any(len(review) > 10_000 for review in reviews):
        raise HTTPException(
            status_code=422, detail="Each review must be 10,000 characters or fewer."
        )
    results = predict_batch(reviews, request.model)
    return {"results": results}


@app.post("/api/train", status_code=202)
def start_training(request: TrainingRequest) -> dict[str, str]:
    global TRAINING_JOB
    with TRAINING_LOCK:
        if TRAINING_JOB is not None:
            process = TRAINING_JOB["process"]
            if isinstance(process, subprocess.Popen) and process.poll() is None:
                raise HTTPException(
                    status_code=409, detail="Training is already in progress."
                )

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex
        log_path = ARTIFACTS_DIR / "training.log"
        command = [
            sys.executable,
            "-u",
            str(PROJECT_DIR / "train.py"),
            "--epochs",
            str(request.epochs),
            "--batch-size",
            str(request.batch_size),
            "--learning-rate",
            str(request.learning_rate),
            "--data-dir",
            str(PROJECT_DIR / "data"),
            "--output-dir",
            str(ARTIFACTS_DIR),
        ]
        if request.max_train_samples is not None:
            command.extend(["--max-train-samples", str(request.max_train_samples)])
        if request.max_test_samples is not None:
            command.extend(["--max-test-samples", str(request.max_test_samples)])

        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_DIR,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
        except OSError as error:
            LOGGER.exception("Could not start model training.")
            raise HTTPException(
                status_code=500, detail=f"Could not start training: {error}"
            ) from error

        TRAINING_JOB = {
            "id": job_id,
            "process": process,
            "log_path": log_path,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    return {"job_id": job_id, "status": "running"}


@app.get("/api/train/{job_id}")
def training_status(job_id: str) -> dict[str, object]:
    with TRAINING_LOCK:
        job = TRAINING_JOB
        if job is None or job["id"] != job_id:
            raise HTTPException(status_code=404, detail="Training job not found.")
        process = job["process"]
        log_path = job["log_path"]
        if not isinstance(process, subprocess.Popen) or not isinstance(
            log_path, Path
        ):
            raise HTTPException(status_code=500, detail="Training job state is invalid.")
        exit_code = process.poll()
        log = log_path.read_text(encoding="utf-8", errors="replace")[-12_000:]

    return {
        "job_id": job_id,
        "status": "running" if exit_code is None else "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "log": log,
    }
