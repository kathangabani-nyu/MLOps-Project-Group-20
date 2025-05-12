"""
FastAPI application for model serving.
Provides health check and prediction endpoints.
"""

from fastapi import FastAPI, Query
import uvicorn
import torch
import os

app = FastAPI()

# Simulate loading a model (replace with actual model loading in production)
MODEL_PATH = "model.pth"
model_loaded = os.path.exists(MODEL_PATH)

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    return {"status": "healthy", "model_loaded": model_loaded}

@app.get("/predict")
async def predict(input_value: float = Query(..., description="Input value for prediction")):
    """
    Simulate a model prediction.
    """
    if not model_loaded:
        return {"error": "Model not loaded."}
    # Simulate a prediction (replace with actual model inference)
    prediction = input_value * 2.5  # Dummy logic for demonstration
    return {"input": input_value, "prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
