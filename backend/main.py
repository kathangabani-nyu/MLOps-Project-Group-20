from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import BartTokenizerFast, BartForConditionalGeneration, pipeline
from peft import PeftModel
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app and enable CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten this in production!
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class TextRequest(BaseModel):
    text: str

# Paths to your vendored model files (base weights) and adapter files
BASE_DIR    = "./models/base"   # mounted from /mnt/block/bart-large-cnn
ADAPTER_DIR = "./models"        # your LoRA adapter files live here

# Load tokenizer & base BART model from local files only
print(f"Loading tokenizer from {BASE_DIR} …")
tokenizer = BartTokenizerFast.from_pretrained(BASE_DIR, local_files_only=True)

print(f"Loading base model from {BASE_DIR} …")
base_model = BartForConditionalGeneration.from_pretrained(
    BASE_DIR,
    local_files_only=True
)

# Wrap base model + LoRA adapters
print(f"Loading LoRA adapters from {ADAPTER_DIR} …")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
    local_files_only=True
)
model.eval()

# Create a summarization pipeline
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=device,
    framework="pt"
)
print("Model loaded")

# Summarization endpoint
@app.post("/summarize")
async def summarize(req: TextRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    result = summarizer(
        text,
        max_length=150,
        min_length=40,
        do_sample=False
    )
    return {"summary": result[0]["summary_text"]}

# Run with `uvicorn main:app --host 0.0.0.0 --port 3500`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

