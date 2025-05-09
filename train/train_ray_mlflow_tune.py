#!/usr/bin/env python

"""
train_bart_ray_mlflow_tune.py

Distributed fine-tuning of facebook/bart-large-cnn on CNN-DailyMail
using Ray Tune for hyperparameter sweeps, Ray Train (TorchTrainer) for
scaling, and MLflow + MinIO for experiment tracking & artifact storage.

Usage:
  # On chi@tacc head node (outside of containers):
  # ray start --head --port=6379 --num-gpus=2 --num-cpus=16 --dashboard-port=8265
  # On chi@tacc worker nodes/containers:
  # ray start --address 127.0.0.1:6379 --num-gpus=1 --num-cpus=8 --block
  # Launch hyperparameter tuning:
  python train_bart_ray_mlflow_tune.py
"""
import os
import mlflow
import torch
from datasets import load_dataset
from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed
)
from peft import get_peft_model, LoraConfig, TaskType

import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from ray.train import RunConfig
from ray.tune.integration.mlflow import MLflowLoggerCallback

# -----------------------------------------------------------------------------
# 1) MLflow + MinIO configuration (on KVM@TACC)
# -----------------------------------------------------------------------------
KVM_IP = os.environ.get("KVM_IP")
if not KVM_IP:
    raise RuntimeError("Environment variable KVM_IP not set. Export KVM_IP=<kvm@tacc_IP> before running.")

os.environ["MLFLOW_TRACKING_URI"]    = f"http://{KVM_IP}:8000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{KVM_IP}:9000"
# MinIO credentials (ensure match kvm@tacc setup)
os.environ["AWS_ACCESS_KEY_ID"]      = os.environ.get("AWS_ACCESS_KEY_ID", "Project20")
os.environ["AWS_SECRET_ACCESS_KEY"]  = os.environ.get("AWS_SECRET_ACCESS_KEY", "Project20")

# Directory to store Tune artifacts locally (optional override)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/tune_results")

# -----------------------------------------------------------------------------
# 2) Hyperparameter search space & scheduler
# -----------------------------------------------------------------------------
search_space = {
    "batch_size": tune.choice([2, 4]),
    "learning_rate": tune.loguniform(1e-5, 5e-5),
    "lora_r": tune.choice([8, 16, 32]),
    "grad_accum": tune.choice([4, 8])
}
scheduler = ASHAScheduler(
    metric="eval_loss",
    mode="min",
    max_t=3,
    grace_period=1,
    reduction_factor=2
)

# -----------------------------------------------------------------------------
# 3) Training function for Tune
# -----------------------------------------------------------------------------
def train_tune(config):
    # Determine Ray world size and rank
    world_size = train.world_size()
    rank = train.world_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42 + rank)

    # Load BART + apply LoRA
    model_id = "facebook/bart-large-cnn"
    tokenizer = BartTokenizerFast.from_pretrained(model_id)
    base_model = BartForConditionalGeneration.from_pretrained(model_id)
    lora_cfg = LoraConfig(
        r=config["lora_r"], lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(base_model, lora_cfg).to(device)
    model.train()

    # Shard dataset across Ray workers
    raw = load_dataset("cnn_dailymail", "3.0.0", split="train")
    ds = raw.shard(num_shards=world_size, index=rank)
    ds = ds.select(range(500 // world_size))

    def preprocess(batch):
        inp = tokenizer(batch["article"], truncation=True, padding="max_length", max_length=1024)
        with tokenizer.as_target_tokenizer():
            lbl = tokenizer(batch["highlights"], truncation=True, padding="max_length", max_length=128)
        inp["labels"] = lbl["input_ids"]
        return inp

    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    collator = DataCollatorForSeq2Seq(tokenizer, model)

    # Seq2Seq training arguments
    args = Seq2SeqTrainingArguments(
        output_dir="/tmp/tune_output",
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["learning_rate"],
        num_train_epochs=3,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=False,
        report_to="none",
        ddp_find_unused_parameters=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=ds,
        data_collator=collator
    )

    trainer.train()
    metrics = trainer.evaluate()

    # Checkpoint & report to Ray Tune
    ckpt_dir = "/tmp/checkpoint"
    trainer.save_model(ckpt_dir)
    train.report(
        {"eval_loss": metrics.get("eval_loss")},
        checkpoint=Checkpoint.from_directory(ckpt_dir)
    )

# -----------------------------------------------------------------------------
# 4) Launch hyperparameter tuning with Ray Tune
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Connect to Ray cluster (chi@tacc head)
    ray_address = os.environ.get("RAY_ADDRESS", "127.0.0.1:6379")
    ray.init(address=ray_address, _node_ip_address="0.0.0.0")

    tuner = tune.Tuner(
        tune.with_resources(
            train_tune,
            resources={"cpu": 4, "gpu": 1}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=8,
            metric="eval_loss",
            mode="min"
        ),
        run_config=RunConfig(
            name="bart_hpo",
            storage_path=OUTPUT_DIR,
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
                    experiment_name="transcept-summarization-hpo",
                    save_artifact=True
                )
            ]
        )
    )

    results = tuner.fit()
    best = results.get_best_result(metric="eval_loss", mode="min")
    print("Best config found:", best.config)
    print("Best checkpoint:", best.checkpoint)

    # -----------------------------------------------------------------------------
    # 5) Register the best model in MLflow
    # -----------------------------------------------------------------------------
    best_model_dir = best.checkpoint.to_directory()
    mlflow.pytorch.log_model(
        torch.load(os.path.join(best_model_dir, "pytorch_model.bin")),
        artifact_path="best_model",
        registered_model_name="BartCNN_LoRA_HPO"
    )
    mlflow.end_run()
