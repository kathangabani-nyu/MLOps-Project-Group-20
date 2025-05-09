import os
import time
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PEFT_DIR   = "./models/peft_checkpoint"
BASE_ID    = "facebook/bart-large-cnn"
DEVICE     = torch.device("cpu")
BATCH_SIZE = 8
WARMUP     = 10
ITERS      = 50
MAX_LEN    = 1024            # BART-CNN’s max positional embeddings

# ── 1) MODEL SIZE ───────────────────────────────────────────────────────────────
adapter_file = os.path.join(PEFT_DIR, "adapter_model.safetensors")
size_mb = os.path.getsize(adapter_file) / (1024*1024)
print(f"1) Adapter size on disk: {size_mb:.2f} MB\n")

# ── 2) LOAD MODEL & TOKENIZER ───────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
base      = AutoModelForSeq2SeqLM.from_pretrained(BASE_ID).to(DEVICE)
model     = base  # if you’re using a PEFT wrapper, wrap here instead
model.eval()

# ── 3) TIMING HELPERS ───────────────────────────────────────────────────────────
def measure_latency(mdl, tok_inputs):
    # warm-up
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = mdl.generate(**tok_inputs, max_new_tokens=50, min_length=1)
    # timed
    times = []
    with torch.no_grad():
        for _ in range(ITERS):
            t0 = time.perf_counter()
            _ = mdl.generate(**tok_inputs, max_new_tokens=50, min_length=1)
            t1 = time.perf_counter()
            times.append(t1 - t0)
    arr = np.array(times)
    return arr.mean()*1e3, np.median(arr)*1e3

def measure_throughput(mdl, tok_inputs):
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = mdl.generate(**tok_inputs, max_new_tokens=50, min_length=1)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(ITERS):
            _ = mdl.generate(**tok_inputs, max_new_tokens=50, min_length=1)
    t1 = time.perf_counter()
    total_tokens = tok_inputs["input_ids"].shape[0] * ITERS
    return total_tokens / (t1 - t0)

# ── 4) SINGLE-SAMPLE LATENCY ────────────────────────────────────────────────────
sample_text = "The quick brown fox jumps over the lazy dog."
single = tokenizer(
    sample_text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LEN,
    padding="max_length"
).to(DEVICE)

mean_lat, med_lat = measure_latency(model, single)
print(f"2) Single-sample latency: mean {mean_lat:.1f} ms, median {med_lat:.1f} ms")

# ── 5) BATCH THROUGHPUT ────────────────────────────────────────────────────────
batch_texts = [sample_text]*BATCH_SIZE
batch = tokenizer(
    batch_texts,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LEN,
    padding="max_length"
).to(DEVICE)

tp = measure_throughput(model, batch)
print(f"3) Batch throughput (@{BATCH_SIZE} samples): {tp:.1f} samples/sec\n")

# ── 6) TEST “ACCURACY” VIA ROUGE ───────────────────────────────────────────────
ds      = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{BATCH_SIZE*2}]")
scorer  = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
metrics = ["rouge1","rouge2","rougeL"]
scores  = {m: [] for m in metrics}

for ex in ds:
    inp = tokenizer(
        ex["article"],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    ).to(DEVICE)
    out  = model.generate(**inp, max_new_tokens=50, min_length=1)
    pred = tokenizer.decode(out[0], skip_special_tokens=True)
    ref  = ex["highlights"].strip()

    result = scorer.score(ref, pred)
    for m in metrics:
        scores[m].append(result[m].fmeasure)

print("4) Test ROUGE (mean f-measure):")
for m in metrics:
    print(f"   {m}: {np.mean(scores[m])*100:.2f}%")
