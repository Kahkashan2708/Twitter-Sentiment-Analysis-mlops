import os
import pickle
import time
import re
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Twitter Sentiment API")

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
PREDICTIONS = Counter('api_predictions_total', 'Number of predictions', ['model', 'label'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency', ['endpoint'])

# Smart path detection (works both locally and in Docker)
if os.path.exists("/app/models"):
    MODELS_DIR = "/app/models"
    VECTORIZER_PATH = "/app/data/processed/vectorizer.pkl"
else:
    MODELS_DIR = "models"
    VECTORIZER_PATH = "data/processed/vectorizer.pkl"

MODEL_FILES = {
    "logistic": os.path.join(MODELS_DIR, "logisticregression.pkl"),
    "lightgbm": os.path.join(MODELS_DIR, "lightgbm.pkl"),
}

# Load vectorizer
vectorizer = None
if os.path.exists(VECTORIZER_PATH):
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print(f"[api] Loaded vectorizer from {VECTORIZER_PATH}")
else:
    print(f"[api] Warning: {VECTORIZER_PATH} not found. Vectorizer unavailable.")

# Load models
loaded_models = {}
for name, path in MODEL_FILES.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            loaded_models[name] = pickle.load(f)
        print(f"[api] Loaded {name} model from {path}")
    else:
        print(f"[api] Warning: {path} not found. {name} model unavailable.")

def preprocess_text(text):
    """Simple text preprocessing"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class PredictRequest(BaseModel):
    text: str
    model: Optional[str] = "logistic"

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "models_loaded": list(loaded_models.keys()),
        "vectorizer_loaded": vectorizer is not None
    }

@app.post("/predict")
def predict(req: PredictRequest):
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()

    if vectorizer is None:
        raise HTTPException(status_code=500, detail="Vectorizer not loaded")

    model_name = req.model.lower()
    if model_name not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available.")

    model = loaded_models[model_name]
    try:
        # Preprocess and vectorize text
        cleaned_text = preprocess_text(req.text)
        X = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(X)[0]
        label = "positive" if prediction == 1 else "negative"
        
        PREDICTIONS.labels(model=model_name, label=label).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

        return {"model": model_name, "prediction": int(prediction), "label": label, "text": req.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
# sdfdsdsgds