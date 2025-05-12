Transcept – LoRA‑Fine‑Tuned Summarization & Q‑A for Lecture Transcripts

Table of Contents
Background & Value Proposition


Quick Start (10‑min demo)


Repository Layout


Data Pipeline (ETL → Feature Store)


Model Training & Retraining Workflows


Serving Architecture & API


Monitoring & Dashboards


CI/CD & Dev Workflow


Running on Chameleon


Evaluation Results


Contributors



Background & Value Proposition
Long STEM lectures bury the handful of concepts students need for assignments and exams. Transcept automatically:
Transcribes lecture Video or Audio with Whisper.


Summarises the transcript into a 70 % shorter bullet list via a LoRA‑fine‑tuned Mistral‑7B‑Instruct.


Answers questions on‑demand with a LoRA‑fine‑tuned Phi‑3.5 Mini RAG pipeline(To be Implemented).



Quick Start (10‑min demo)
# clone & enter project
$ git clone https://github.com/kathangabani-nyu/MLOps-Project-Group-20

# spin up everything locally (CPU only)
$ docker compose up --build -d  # Building UI and FAST API along with the Model

NOTE GPU inference & full dashboards require the Chameleon deployment described below.

LINK TO THE UI : http://129.114.25.36:5000/

Repository Layout
Path
Purpose
Airflow/
DAG definitions – nightly eval & drift‑based retrain
Configuration/
YAMLs & hyper‑param files (see mistral_lora.yaml)
Data/
ETL & preprocessing scripts (e.g. audio_transcriber.py)
Docker/
Dockerfiles (Jupyter, Ray, ROCm) + docker-object-store.yaml
Evaluation/
ROUGE & latency eval code
Inference Performance/
Load‑testing notebooks (PyTorch_on_CPU.py)
Monitoring/
Prometheus exporters & FastAPI probes
Storage/
Optional – local object‑store setup scripts
backend/
FastAPI implementation – see main.py
dashboard/
Grafana & business dashboards – e.g. business_dashboard.py
frontend/
Simple React UI (requirements in requirements.txt)
infrastructure/
Chameleon launch scripts – launch_k8s_cluster.py
train/
Ray + PyLightning LoRA fine‑tunes – train_model2_ray.py
docker-compose.yml
One‑shot local deployment


Data Pipeline (ETL → Feature Store)
Raw Lecture Audio & Video → stored in the MinIO bucket raw-inputs (mounted at /mnt/object).


Transcription – Frontend/app.py wraps Whisper‑large‑v3. Word‑error‑rate (WER) stats logged to Prometheus.


Cleaning & Segmentation – Storage/create_server.ipynb removes stopwords, normalises Unicode, and chunks text into ≤512 token windows.


Dataset Prep – Data Jason files aligns CNN/DailyMail reference summaries and labels each chunk.


Artifacts are versioned in the MinIO bucket preprocessed, path‐convention <lecture_id>/<stage>.parquet.
Data Dashboard : We provide an interactive Streamlit-based Dashboard  that reads directly from our block storage on kvm@tacc (mounted at /mnt/block/MLOps-Project-Group-20/data) and gives our team rapid, visual feedback on the raw train/validation/test JSONL datasets before any model training or deployment.



Model Training & Retraining Workflows

Transcept which focuses on scalable model training and infrastructure for lecture summarization and QA using Chameleon Cloud. It demonstrates our ability to train and re-train models, implement large-scale training strategies, utilize distributed training, and manage training infrastructure with experiment tracking and job scheduling.
Model Training at Scale
Modeling
Summarization:
Model: facebook/bart-large-cnn
Inputs: News article text (from CNN/DailyMail dataset)
Outputs: A summary of the article
Rationale: BART is a strong baseline for abstractive summarization with good performance on CNN/DailyMail.
Question Answering:
Model: microsoft/phi-3.5-mini-instruct
Inputs: Question + context (summary)
Outputs: Answer string
Rationale: Phi-3.5 Mini provides efficient inference and performance for instruction-following QA with low VRAM.
Train and Re-train
We implemented two training scripts per task:
Standard SFT (Standalone) 
Distributed Ray Cluster
Hyperparameter tuning 
Retraining is achieved by reusing the same pipeline and updating datasets under /mnt/block and re-running jobs.
Large Model Strategies
Used LoRA (PEFT) with r=16/32 to reduce fine-tuning memory footprint.
Enabled bfloat16 where supported.
Used gradient accumulation (8 steps for BART, 4 for Phi-3.5) to simulate large batch sizes on limited GPU.
Distributed Training Experiments
Experiment Setup:
Dataset: CNN/DailyMail (500 train, 100 eval), SQuAD v2 (10k train, 2k eval)
Batch Size: 2-4
GPU Count: 1 vs 2
Training Time vs. GPUs (BART):
GPUs
Strategy
Time (approx)
1
SFT
41m
2
DDP
22m

Training Time vs. GPUs (Phi-3.5 Mini):
GPUs
Strategy
Time (approx)
1
SFT
54m
2
DDP
30m

Ray Train
Used TorchTrainer for training
ScalingConfig defined with num_workers=2, use_gpu=True
Supported checkpointing and fault tolerance using TorchCheckpoint
Ray Tune
 batch_size, grad_accum, lora_r, learning_rate, precision
Scheduler: ASHAScheduler with grace_period=1
Best config printed + registered in MLflow

Model Training Infrastructure and Platform
Experiment Tracking
Hosted MLflow server in containerized setup via docker-compose-ray.yaml
Artifacts stored in MinIO bucket: mlflow-artifacts
Backend DB: Postgres
Each training script logs:
Metrics (eval_loss, rouge, accuracy, etc.)
Model artifacts (tokenizer, checkpoints)
Environment (MLFLOW_TRACKING_URI, AWS_ACCESS_KEY_ID, etc.)
Scheduling Training Jobs
Training jobs submitted to Ray Cluster via:
ray job submit --runtime-env ray_runtime.json --working-dir . -- python train_phi3p5_ray_mlflow.py
Jobs tracked in Ray Dashboard (8265) + MLflow (8000)
Docker & Runtime Environment
Base Docker image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
Custom Dockerfile.jupyter-ray for Jupyter-based dev and job submission
Ray and MLflow installed in all containers
Runtime defined in ray_runtime.json, including pip requirements and env vars
Distributed Platform Setup
Ray cluster setup 
Mounted /mnt/block1 for dataset access


Serving Architecture & API

FastAPI backend (backend/main.py) exposes: 
POST /summarize
    ↳ request body: { "text": "<raw transcript or paragraph>" }
    ↳ response:     { "summary": "<150‑token abstractive summary>" }
Model stack
BartTokenizerFast + BartForConditionalGeneration (base weights vendored in ./models/base).
LoRA adapters (rank 64) loaded via peft.PeftModel from ./models.
Inference runs on GPU cuda:0 when available, else CPU fallback.
Pipeline construction
summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    framework="pt"
)
Deployment
Served by Uvicorn: uvicorn main:app --host 0.0.0.0 --port 8000.
CORS is open to all origins for demo; restrict in production.
Python logging set to INFO for request & model‑load diagnostics.
Reverse proxy & scaling
An Nginx sidecar (see backend/deployment/fastapi_gpu.yaml) terminates TLS and proxies /api traffic to Uvicorn.
HorizontalPodAutoscaler scales the pod based on GPU utilisation (>60 %).


Deployment descriptors live in docker-compose.yml that brings up the frontend and the backend services 

Planned extensions
/transcribe – Whisper ASR endpoint (work‑in‑progress in backend/asr.py).
/qa – Phi‑3.5 Mini Q‑A endpoint (to be added post‑MVP).

CI/CD & Dev Workflow
Stage
Tool
Trigger
Build
GitHub Actions (.github/workflows/ci.yaml)
PR push
Image Publish
Docker Hub
Build success
Deploy
Argo CD (backend/staging_pipeline.yaml)
Image tag main-*
Canary + Rollback
Argo metrics (argo-rollouts.yaml)
ROUGE‑L regression

Dev containers for VS Code are provided in Docker/dev.

 Airflow Pipeline (Kathan)
DAG: airflow/dags/audio_ml_pipeline.py


ONNX helpers: airflow/dags/helpers/onnx_utils.py


export_onnx: /mnt/block/model1-artifacts/model.pth → onnx_models/model.onnx


optimize_onnx: onnx_models/model.onnx → onnx_models/optimized/model.opt.onnx


quantize_onnx: onnx_models/optimized/model.opt.onnx → onnx_models/quantized/model.quant.onnx


Backend test scripts:


backend/test_model.py (accuracy/unit tests)


backend/load_test.py (load/performance testing)


backend/monitor.py (system health checks)


Container build & deploy tasks:


build_container → Docker image build


deploy_staging → staging environment


deploy_canary → canary rollout


deploy_prod → production rollout



 Monitoring
Prometheus panels: MLOps-Project-Group-20/Docker/prometheus.yml


Grafana panels: MLOps-Project-Group-20/Docker/grafana.yml



 Retraining Trigger
Fires on data drift or accuracy drop


Automates promotion: staging → canary → production


 Running on Chameleon
Launch VM


VM_setup.ipynb

Deploy Services with Docker Compose
cd airflow
docker compose -f docker-compose.yaml up -d --build


🔗 Service Endpoints (Floating IP: 129.114.25.36)
 Airflow UI: http://129.114.25.36:8080


Jupyter Notebook: http://129.114.25.36:8888


MLflow Tracking: http://129.114.25.36:8000


 FastAPI Backend: http://129.114.25.36:3500


Streamlit Frontend: http://129.114.25.36:5000


Grafana Dashboards: http://129.114.25.36:3000

Deploy object‑store (one‑liner Helm chart)

 $ helm repo add minio https://charts.min.io/
$ helm install minio minio/minio -f Docker/docker-object-store.yaml


Build & push images (if you changed code)

 $ docker compose build
$ docker compose push  # requires registry creds


Deploy micro‑services

 # create namespaces & secrets
$ kubectl apply -f backend/deployment/namespaces.yaml

# deploy FastAPI + workers
$ kubectl apply -f backend/deployment/fastapi_gpu.yaml

# deploy monitoring stack
$ kubectl apply -k Monitoring/kustomize/


Expose endpoints

 $ kubectl get svc -n transcept
NAME            TYPE           EXTERNAL-IP   PORT(S)
fastapi-svc     LoadBalancer   <floating‑ip> 9002:...   
grafana-svc     LoadBalancer   <floating‑ip> 3000:...


Tear down

 $ chameleon delete --cluster transcept



Contributors
Name
Role


Aishwarya
Model Training & Experimentation


Bharath
Serving & Monitoring


Anushka
Data Pipeline & Online Eval


Kathan
CI/CD & Infrastructure



