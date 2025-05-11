from fastapi import FastAPI
import uvicorn
import os

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/predict")
async def predict():
    # Placeholder for model prediction
    return {"prediction": "model not yet implemented"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
