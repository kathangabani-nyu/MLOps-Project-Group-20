#!/usr/bin/env python3
import os
import time
import json
import argparse
import threading
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def load_session(onnx_path):
    """Load ONNX Runtime session."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess

def summarize(sess, tokenizer, text_inputs, max_len):
    """Run a single ONNX inference (returns decoded summary)."""
    inputs = tokenizer(
        text_inputs,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=max_len
    )
    out = sess.run(
        None,
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    )
    # assume first output is logits; take argmax
    ids = np.argmax(out[0], axis=-1)
    # we drop the decoded string here, since we're only measuring latency
    return

def worker(thread_id, sess, tokenizer, samples, max_len, iterations, results):
    """Each thread runs `iterations` inferences cycling through `samples`."""
    latencies = []
    for i in range(iterations):
        text = samples[i % len(samples)]
        t0 = time.perf_counter()
        summarize(sess, tokenizer, text, max_len)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e3)
    results[thread_id] = latencies

def main(args):
    # Load model + tokenizer
    sess      = load_session(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Prepare sample inputs (from first N articles of CNN/DailyMail)
    from datasets import load_dataset
    ds = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{args.batch_size}]")
    samples = [ex["article"] for ex in ds]

    # Spawn worker threads
    threads = []
    results = {}
    start_all = time.perf_counter()
    for tid in range(args.concurrency):
        t = threading.Thread(
            target=worker,
            args=(tid, sess, tokenizer, samples, args.max_len, args.iterations, results)
        )
        t.start()
        threads.append(t)

    # Join
    for t in threads:
        t.join()
    end_all = time.perf_counter()

    # Collate
    all_lat = np.concatenate(list(results.values()))
    total_requests = args.concurrency * args.iterations
    elapsed = end_all - start_all

    metrics = {
        "concurrency": args.concurrency,
        "iterations_per_thread": args.iterations,
        "total_requests": total_requests,
        "elapsed_seconds": elapsed,
        "throughput_req_per_sec": total_requests / elapsed,
        "latency_mean_ms": float(np.mean(all_lat)),
        "latency_p50_ms": float(np.percentile(all_lat, 50)),
        "latency_p90_ms": float(np.percentile(all_lat, 90)),
        "latency_p99_ms": float(np.percentile(all_lat, 99)),
    }

    print(json.dumps(metrics, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved load‐test metrics to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ONNX Summarization Load Test")
    p.add_argument(
        "--model", default=os.getenv("ONNX_MODEL_PATH","serving_models/model.onnx"),
        help="Path to the ONNX model file"
    )
    p.add_argument(
        "--tokenizer", default=os.getenv("TOKENIZER_ID","facebook/bart-large-cnn"),
        help="HF tokenizer ID"
    )
    p.add_argument(
        "--concurrency", type=int, default=4,
        help="Number of concurrent threads"
    )
    p.add_argument(
        "--iterations", type=int, default=50,
        help="Number of inferences per thread"
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of distinct sample inputs to cycle through"
    )
    p.add_argument(
        "--max-len", type=int, default=1024,
        help="Max token length for inputs"
    )
    p.add_argument(
        "--output", default="load_test_metrics.json",
        help="File to write throughput/latency metrics"
    )
    args = p.parse_args()
    main(args)
