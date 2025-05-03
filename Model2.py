#!/usr/bin/env python
"""
train_phi3_ray_mlflow.py
--------------------------------
LoRA fine‑tunes microsoft/Phi‑3‑mini‑4k‑instruct on SQuAD v2
with Ray Train (multi‑GPU) + MLflow tracking / registry.
"""

import argparse, os, logging, json
from datetime import datetime
import mlflow
from datasets import load_dataset, Dataset as HFDataset
import torch

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, set_seed
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ------------- Ray -------------
import ray
from ray import train
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer
# --------------------------------

# ---------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, default=1, help="#GPUs Ray uses")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--lora_r", type=int, default=16, help="0 = full FT")
    p.add_argument("--output_dir", default="phi3_squad_ft")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()
# --------------------------------


def build_datasets(seed=42):
    raw = load_dataset("squad_v2")

    def fmt(ex):
        answer = ex["answers"]["text"][0] if ex["answers"]["text"] else "No answer"
        return {
            "instruction": "Answer the question based on the context:",
            "input": f"Context: {ex['context']}\nQuestion: {ex['question']}",
            "output": answer,
        }

    train_ds = raw["train"].map(fmt).shuffle(seed=seed).select(range(10000))
    val_ds   = raw["validation"].map(fmt).shuffle(seed=seed).select(range(2000))
    return train_ds, val_ds


def ray_train_loop(cfg):
    set_seed(cfg["seed"])
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("transcept-qa")
    mlflow.start_run(run_id=cfg["run_id"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if (cfg["precision"] == "bf16" and
                               torch.cuda.is_bf16_supported()) else torch.float16
    attn_impl = "flash_attention_2" if dtype == torch.bfloat16 else "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map="auto",
    )

    if cfg["lora_r"] > 0:
        lora_cfg = LoraConfig(
            r=cfg["lora_r"], lora_alpha=16, lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
              "k_proj","q_proj","v_proj","o_proj",
              "gate_proj","down_proj","up_proj"]
        )
        model = get_peft_model(model, lora_cfg)

    train_raw, val_raw = build_datasets(cfg["seed"])

    def to_chat(row):
        msgs = [
          {"role": "user",
           "content": f"{row['instruction']}\n{row['input']}"},
          {"role": "assistant", "content": row["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(
            msgs, add_generation_prompt=False, tokenize=False)}

    train_ds = train_raw.map(to_chat, remove_columns=train_raw.column_names)
    val_ds   = val_raw.map(to_chat,   remove_columns=val_raw.column_names)
    train_hf = HFDataset.from_pandas(train_ds.to_pandas())
    val_hf   = HFDataset.from_pandas(val_ds.to_pandas())

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|assistant|>", tokenizer=tokenizer
    )

    args = TrainingArguments(
        output_dir           = cfg["output_dir"],
        per_device_train_batch_size = cfg["batch"],
        per_device_eval_batch_size  = cfg["batch"],
        gradient_accumulation_steps = cfg["grad_accum"],
        learning_rate        = 1e-4,
        num_train_epochs     = cfg["epochs"],
        logging_strategy     = "epoch",
        eval_strategy        = "epoch",
        save_strategy        = "epoch",
        seed                 = cfg["seed"],
        fp16                 = (dtype==torch.float16),
        bf16                 = (dtype==torch.bfloat16),
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_hf,
        eval_dataset    = val_hf,
        data_collator   = collator,
    )

    if cfg["resume_from"]:
        trainer.train(resume_from_checkpoint=cfg["resume_from"])
    else:
        trainer.train()

    metrics = trainer.evaluate(max_length=32, num_beams=4)
    mlflow.log_metrics(metrics)

    ckpt_dir = os.path.join(cfg["output_dir"], "final")
    trainer.save_model(ckpt_dir); tokenizer.save_pretrained(ckpt_dir)
    mlflow.pytorch.log_model(trainer.model, artifact_path="model")

    mlflow.end_run()
    return Checkpoint.from_directory(ckpt_dir)


# ===================== MAIN =========================
def main():
    a = parse_args()

    run = mlflow.start_run(run_name=f"phi3_ray_{datetime.now():%Y%m%d_%H%M%S}")
    run_id = run.info.run_id
    mlflow.end_run()

    cfg = {
        "model_id"   : "microsoft/Phi-3-mini-4k-instruct",
        "batch"      : a.batch,
        "grad_accum" : a.grad_accum,
        "epochs"     : a.epochs,
        "precision"  : a.precision,
        "lora_r"     : a.lora_r,
        "output_dir" : a.output_dir,
        "resume_from": a.resume_from,
        "seed"       : a.seed,
        "run_id"     : run_id,
    }

    ray.init()
    trainer = TorchTrainer(
        train_loop_per_worker = ray_train_loop,
        scaling_config = ScalingConfig(
            num_workers=a.gpus,
            use_gpu=True,
            resources_per_worker={"GPU":1},
        ),
        train_loop_config = cfg,
        run_config = train.RunConfig(storage_path=cfg["output_dir"])
    )
    res = trainer.fit()
    print("Best checkpoint:", res.checkpoint.uri)
    ray.shutdown()


if __name__ == "__main__":
    main()
