"""
main.py
=======
FastAPI REST API — Twitter Sentiment Analysis (Simple RNN).

Loads the trained RNN model + tokenizer from ml_pipeline.py at startup.
All heavy loading happens ONCE — every request reuses the same objects.

Run locally:
    uvicorn main:app --reload --port 8000

Interactive docs (Swagger UI):
    http://localhost:8000/docs

Test single prediction:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "omg its already 7:30 :O"}'

Test batch:
    curl -X POST http://localhost:8000/batch \
         -H "Content-Type: application/json" \
         -d '{"texts": ["I love this!", "This is terrible..."]}'
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_pipeline import load_artifacts, predict_sentiment

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Global store: model + tokenizer + config loaded once at startup ───────────
STORE: dict = {}


# ── Lifespan: load on startup, release on shutdown ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting API — loading RNN model, tokenizer, config...")
    try:
        model, tokenizer, config = load_artifacts()
        STORE["model"]     = model
        STORE["tokenizer"] = tokenizer
        STORE["config"]    = config
        log.info(
            f"RNN ready  |  vocab={config['vocab_size']}  "
            f"max_len={config['max_len']}  embed={config['embed_dim']}"
        )
    except FileNotFoundError as e:
        log.error(str(e))
    yield
    STORE.clear()
    log.info("Shutdown — model released")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Twitter Sentiment API  (Simple RNN)",
    description = (
        "Predicts **positive** or **negative** sentiment from tweet text.\n\n"
        "**Model architecture:**\n"
        "```\n"
        "Embedding(20000, 64)\n"
        "  → SimpleRNN(64, tanh)\n"
        "  → Dropout(0.3)\n"
        "  → Dense(1, sigmoid)\n"
        "```\n"
        "Trained on the Twitter Sentiment dataset "
        "(columns: ItemID, Sentiment, SentimentText)."
    ),
    version = "2.0.0",
    lifespan = lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length = 2,
        max_length = 280,
        example    = "omg its already 7:30 :O",
    )

class PredictResponse(BaseModel):
    sentiment:     str    # "positive" | "negative"
    confidence:    float  # probability of predicted class
    probabilities: dict   # {"negative": 0.13, "positive": 0.87}
    text_preview:  str    # first 80 chars of input
    processing_ms: float  # inference time


class BatchRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length = 1,
        max_length = 50,
        example    = [
            "is so sad for my APL friend............",
            "Juuuuuuussst Chillin!!",
        ],
    )

class BatchItem(BaseModel):
    text_preview:  str
    sentiment:     str
    confidence:    float
    probabilities: dict

class BatchResponse(BaseModel):
    results:       list[BatchItem]
    total:         int
    processing_ms: float


class HealthResponse(BaseModel):
    status:        str   # "ok" | "model_not_loaded"
    model_loaded:  bool
    model_type:    str
    version:       str


# ── Helper: check model is ready ──────────────────────────────────────────────
def _require_model():
    if "model" not in STORE:
        raise HTTPException(
            status_code = 503,
            detail      = (
                "RNN model not loaded. "
                "Run:  python ml_pipeline.py --data dataset.csv"
            ),
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    """API overview."""
    return {
        "name":    "Twitter Sentiment API",
        "version": "2.0.0",
        "model":   "Simple RNN (Keras / TensorFlow)",
        "labels":  ["positive", "negative"],
        "endpoints": {
            "health":  "GET  /health",
            "predict": "POST /predict",
            "batch":   "POST /batch",
            "docs":    "GET  /docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """
    Liveness check endpoint.

    Used by:
      - Render.com  — to confirm the app started after deploy
      - GitHub Actions CI/CD  — to verify deployment succeeded
      - AWS ALB health checks  — (next phase)

    Always returns HTTP 200. Check 'model_loaded' field for model status.
    """
    return HealthResponse(
        status       = "ok" if "model" in STORE else "model_not_loaded",
        model_loaded = "model" in STORE,
        model_type   = "SimpleRNN (Keras)",
        version      = "2.0.0",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict sentiment for a **single** tweet or text string.

    **Flow inside the RNN:**
    1. Clean text (remove @, URLs, punctuation)
    2. Tokenize → integer sequence
    3. Pad to MAX_LEN=50 tokens
    4. Pass through Embedding → SimpleRNN → Dense(sigmoid)
    5. Output probability → label

    **Example input:**
    ```json
    {"text": "omg its already 7:30 :O"}
    ```

    **Example output:**
    ```json
    {
      "sentiment":     "positive",
      "confidence":    0.83,
      "probabilities": {"negative": 0.17, "positive": 0.83},
      "text_preview":  "omg its already 7:30 :O",
      "processing_ms": 12.4
    }
    ```
    """
    _require_model()

    t0 = time.perf_counter()
    try:
        result = predict_sentiment(
            STORE["model"],
            STORE["tokenizer"],
            STORE["config"],
            req.text,
        )
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info(
        f"predict | {result['sentiment']} ({result['confidence']:.2f}) "
        f"| {ms}ms | '{req.text[:40]}'"
    )

    return PredictResponse(
        sentiment     = result["sentiment"],
        confidence    = result["confidence"],
        probabilities = result["probabilities"],
        text_preview  = req.text[:80],
        processing_ms = ms,
    )


@app.post("/batch", response_model=BatchResponse, tags=["Prediction"])
def batch_predict(req: BatchRequest):
    """
    Predict sentiment for up to **50 tweets** in a single API call.

    Useful for:
    - Bulk-processing a downloaded timeline
    - Analysing a CSV file of tweets
    - A/B testing multiple text variants

    **Example input:**
    ```json
    {
      "texts": [
        "is so sad for my APL friend............",
        "Juuuuuuussst Chillin!!"
      ]
    }
    ```
    """
    _require_model()

    t0 = time.perf_counter()
    items = []

    for text in req.texts:
        try:
            r = predict_sentiment(
                STORE["model"],
                STORE["tokenizer"],
                STORE["config"],
                text,
            )
        except Exception as e:
            log.warning(f"Batch item error: {e}")
            r = {
                "sentiment":     "error",
                "confidence":    0.0,
                "probabilities": {},
            }
        items.append(BatchItem(
            text_preview  = text[:80],
            sentiment     = r["sentiment"],
            confidence    = r["confidence"],
            probabilities = r.get("probabilities", {}),
        ))

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info(f"batch | {len(items)} items | {total_ms}ms")

    return BatchResponse(
        results       = items,
        total         = len(items),
        processing_ms = total_ms,
    )


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)