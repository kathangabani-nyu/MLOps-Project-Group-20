#!/usr/bin/env python3
import os
import json
import argparse
import logging
import numpy as np
import onnxruntime as ort

from transformers import AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def load_session(onnx_path):
    """Load ONNX Runtime session."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    logging.info("✅ Loaded ONNX model from %s", onnx_path)
    return sess

def summarize(sess, tokenizer, text, max_len):
    """Run the ONNX model to produce a summary string."""
    inputs = tokenizer(
        text,
        return_tensors="np",
        max_length=max_len,
        padding="max_length",
        truncation=True
    )
    output = sess.run(
        None,
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    )
    # assume first output is logits; take argmax
    pred_ids = np.argmax(output[0], axis=-1)
    return tokenizer.decode(pred_ids[0], skip_special_tokens=True)

def compute_general(sess, tokenizer, scorer, dataset_split, batch_size, max_len):
    """Compute average ROUGE on a CNN/DailyMail slice."""
    ds = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{batch_size*2}]")
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ex in ds:
        pred = summarize(sess, tokenizer, ex["article"], max_len)
        ref  = ex["highlights"]
        res  = scorer.score(ref, pred)
        for m in scores:
            scores[m].append(res[m].fmeasure)
    avg = {f"general_{m}": float(np.mean(v)) for m, v in scores.items()}
    return avg

def compute_slices(sess, tokenizer, scorer, ds, max_len):
    """Compute ROUGE1 broken out by article length."""
    slice_scores = {"short": [], "medium": [], "long": []}
    for ex in ds:
        L = len(ex["article"].split())
        cat = "short" if L < 100 else "medium" if L < 300 else "long"
        pred = summarize(sess, tokenizer, ex["article"], max_len)
        r1   = scorer.score(ex["highlights"], pred)["rouge1"].fmeasure
        slice_scores[cat].append(r1)
    return {f"slice_{cat}_rouge1": float(np.mean(vals)) for cat, vals in slice_scores.items()}

def compute_templates(sess, tokenizer, templates, max_len):
    """Run sanity‐check summaries on fixed templates."""
    out = {}
    for name, text in templates.items():
        summary = summarize(sess, tokenizer, text, max_len)
        out[f"templ_{name}"] = summary
    return out

def compute_failures(sess, tokenizer, failures, max_len):
    """Run known failure‐mode prompts through the model."""
    out = {}
    for name, text in failures.items():
        summary = summarize(sess, tokenizer, text, max_len)
        out[f"fail_{name}"] = summary
    return out

def main(args):
    # Load model + tokenizer + scorer
    sess      = load_session(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    scorer    = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

    # 1) General metrics
    gen_metrics = compute_general(
        sess, tokenizer, scorer,
        dataset_split=args.dataset_split,
        batch_size=args.batch_size,
        max_len=args.max_len
    )
    logging.info("General metrics: %s", gen_metrics)

    # 2) Example for human sanity‐check
    ds_example = load_dataset("cnn_dailymail", "3.0.0", split=f"test[{args.batch_size*2}:{args.batch_size*2+1}]")[0]
    example_pred = summarize(sess, tokenizer, ds_example["article"], args.max_len)
    logging.info("Example article excerpt: %s …", ds_example["article"][:200])
    logging.info("Example summary: %s", example_pred)

    # 3) Template tests
    templates = {
        "empty": "",
        "trivial": "Hello world. Summarize this.",
        "long_rep": "Test " * 300
    }
    templ_results = compute_templates(sess, tokenizer, templates, args.max_len)
    logging.info("Template summaries: %s", templ_results)

    # 4) Slices
    # reuse the same ds used for general, but break it out
    ds_slice = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{args.batch_size*2}]")
    slice_metrics = compute_slices(sess, tokenizer, scorer, ds_slice, args.max_len)
    logging.info("Slice metrics: %s", slice_metrics)

    # 5) Known failure modes
    failures = {
        "negation": "It was not bad, yet not great. Summarize.",
        "numerical": "In 2020 GDP grew by 2.3% but slowed in Q4. Summarize."
    }
    fail_results = compute_failures(sess, tokenizer, failures, args.max_len)
    logging.info("Failure summaries: %s", fail_results)

    # 6) Assemble JSON output
    output = {
        **gen_metrics,
        **slice_metrics,
        **templ_results,
        **fail_results,
        # simple pass/fail example
        "test_pass_general_rouge1_gt_0.3": gen_metrics["general_rouge1"] > 0.3
    }

    # Save
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logging.info("Saved evaluation metrics to %s", args.output)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Offline evaluation of ONNX summarization model")
    p.add_argument(
        "--model", default=os.getenv("ONNX_MODEL_PATH","serving_models/model.onnx"),
        help="Path to the ONNX model file"
    )
    p.add_argument(
        "--tokenizer", default=os.getenv("TOKENIZER_ID","facebook/bart-large-cnn"),
        help="Hugging Face tokenizer identifier (to tokenize inputs)"
    )
    p.add_argument(
        "--dataset-split", default="test",
        help="Which split of CNN/DailyMail to evaluate on (e.g. 'test')"
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of samples for general metrics (and for slice bin size)"
    )
    p.add_argument(
        "--max-len", type=int, default=1024,
        help="Max token length for the inputs"
    )
    p.add_argument(
        "--output", default=os.getenv("EVAL_OUTPUT","evaluation_results.json"),
        help="Where to write the JSON metrics"
    )
    args = p.parse_args()
    main(args)
