from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_price, batch_predict
from schemas import HousePredictionRequest, PredictionResponse
from prometheus_client import start_http_server, Counter, Histogram
import threading
import time

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])

# Start Prometheus metrics server on port 9100 in a background thread
def start_metrics_server():
    start_http_server(9100)

# Start metrics server in background thread
metrics_thread = threading.Thread(target=start_metrics_server, daemon=True)
metrics_thread.start()

# Initialize FastAPI app with metadata
app = FastAPI(
    title="House Price Prediction API",
    description="An API for predicting house prices based on various features.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    start_time = time.time()
    result = predict_price(request)
    REQUEST_LATENCY.labels(method='POST', endpoint='/predict').observe(time.time() - start_time)
    REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='200').inc()
    return result

# Batch prediction endpoint
@app.post("/batch-predict", response_model=list)
async def batch_predict_endpoint(requests: list[HousePredictionRequest]):
    start_time = time.time()
    result = batch_predict(requests)
    REQUEST_LATENCY.labels(method='POST', endpoint='/batch-predict').observe(time.time() - start_time)
    REQUEST_COUNT.labels(method='POST', endpoint='/batch-predict', status='200').inc()
    return result