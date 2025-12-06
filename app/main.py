# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict
import os

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# init tracer
provider = TracerProvider()
trace.set_tracer_provider(provider)
# Add exporters - Console for quick debugging; add OTLP exporter if desired
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
# Optionally enable OTLP export if you have a collector / endpoint configured
# otlp_exporter = OTLPSpanExporter(endpoint=os.environ.get("OTLP_ENDPOINT", "http://localhost:4318/v1/traces"))
# provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

tracer = trace.get_tracer(__name__)

app = FastAPI(title="Fraud Detector")

# load model
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
model = joblib.load(MODEL_PATH)

class Transaction(BaseModel):
    # accept dict of features or list; we'll accept any mapping
    features: Dict[str, float]

@app.post("/predict")
async def predict(t: Transaction):
    # convert to dataframe (single row)
    X = pd.DataFrame([t.features])

    # create span that measures model inference time
    with tracer.start_as_current_span("model.predict") as span:
        probs = model.predict_proba(X)[:,1]
        preds = model.predict(X)

    return {
        "prediction": int(preds[0]),
        "probability": float(probs[0])
    }

@app.get("/")
def root():
    return {"status":"ok"}
