![Blank diagram (3)](https://github.com/user-attachments/assets/7a55c086-b0df-45bd-8a0b-ab84f2a85003)
---
base_model: facebook/bart-large-cnn
library_name: peft
---

# Transcept – LoRA‑Fine‑Tuned Summarization & Q‑A for Lecture Transcripts

## Table of Contents
- [Background & Value Proposition](#background--value-proposition)
- [Quick Start (10‑min demo)](#quick-start-10min-demo)
- [Repository Layout](#repository-layout)
- [Data Pipeline (ETL → Feature Store)](#data-pipeline-etl--feature-store)
- [Model Training & Retraining Workflows](#model-training--retraining-workflows)
- [Serving Architecture & API](#serving-architecture--api)
- [Monitoring & Dashboards](#monitoring--dashboards)
- [CI/CD & Dev Workflow](#cicd--dev-workflow)
- [Running on Chameleon](#running-on-chameleon)
- [Contributors](#contributors)

## Background & Value Proposition
Long STEM lectures bury the handful of concepts students need for assignments and exams. Transcept automatically:
- Transcribes lecture Video or Audio with Whisper
- Summarises the transcript into a 70% shorter bullet list via a LoRA‑fine‑tuned Mistral‑7B‑Instruct
- Answers questions on‑demand with a LoRA‑fine‑tuned Phi‑3.5 Mini RAG pipeline (To be Implemented)

## Quick Start (10‑min demo)
```bash
# clone & enter project
git clone https://github.com/kathangabani-nyu/MLOps-Project-Group-20

# spin up everything locally (CPU only)
docker compose up --build -d  # Building UI and FAST API along with the Model
```

> **Note:** GPU inference & full dashboards require the Chameleon deployment described below.

**Live Demo:** [http://129.114.25.36:5000/](http://129.114.25.36:5000/)

## Repository Layout

| Path | Purpose |
|------|---------|
| `Airflow/` | DAG definitions – nightly eval & drift‑based retrain |
| `Configuration/` | YAMLs & hyper‑param files (see mistral_lora.yaml) |
| `Data/` | ETL & preprocessing scripts (e.g. audio_transcriber.py) |
| `Docker/` | Dockerfiles (Jupyter, Ray, ROCm) + docker-object-store.yaml |
| `Evaluation/` | ROUGE & latency eval code |
| `Inference Performance/` | Load‑testing notebooks (PyTorch_on_CPU.py) |
| `Monitoring/` | Prometheus exporters & FastAPI probes |
| `Storage/` | Optional – local object‑store setup scripts |
| `backend/` | FastAPI implementation – see main.py |
| `dashboard/` | Grafana & business dashboards – e.g. business_dashboard.py |
| `frontend/` | Simple React UI (requirements in requirements.txt) |
| `infrastructure/` | Chameleon launch scripts |
| `train/` | Ray + PyLightning LoRA fine‑tunes – train_model2_ray.py |
| `docker-compose.yml` | One‑shot local deployment |

## Data Pipeline (ETL → Feature Store)

1. **Raw Lecture Audio & Video** → stored in the MinIO bucket `raw-inputs` (mounted at `/mnt/object`)
2. **Transcription** – Frontend/app.py wraps Whisper‑large‑v3. Word‑error‑rate (WER) stats logged to Prometheus
3. **Cleaning & Segmentation** – Storage/create_server.ipynb removes stopwords, normalises Unicode, and chunks text into ≤512 token windows
4. **Dataset Prep** – Data Jason files aligns CNN/DailyMail reference summaries and labels each chunk.

Artifacts are versioned in the MinIO bucket `preprocessed`, path‑convention `<lecture_id>/<stage>.parquet`

**Data Dashboard:** Interactive Streamlit-based Dashboard that reads directly from block storage on kvm@tacc (mounted at `/mnt/block/MLOps-Project-Group-20/data`) providing rapid, visual feedback on raw train/validation/test JSONL datasets.
**Demo:** [http://129.114.25.36:5565/](http://129.114.25.36:9002/)
## Model Training & Retraining Workflows

### Model Training at Scale

#### Modeling
**Summarization:**
- Model: `facebook/bart-large-cnn`
- Inputs: News article text (from CNN/DailyMail dataset)
- Outputs: A summary of the article
- Rationale: BART is a strong baseline for abstractive summarization with good performance on CNN/DailyMail

**Question Answering:**
- Model: `microsoft/phi-3.5-mini-instruct`
- Inputs: Question + context (summary)
- Outputs: Answer string
- Rationale: Phi-3.5 Mini provides efficient inference and performance for instruction-following QA with low VRAM

### Training Performance

#### Training Time vs. GPUs (BART)
| GPUs | Strategy | Time (approx) |
|------|----------|---------------|
| 1    | SFT      | 41m          |
| 2    | DDP      | 22m          |

#### Training Time vs. GPUs (Phi-3.5 Mini)
| GPUs | Strategy | Time (approx) |
|------|----------|---------------|
| 1    | SFT      | 54m          |
| 2    | DDP      | 30m          |

### Training Infrastructure
- **Experiment Tracking:** MLflow server in containerized setup
- **Artifacts:** Stored in MinIO bucket: `mlflow-artifacts`
- **Backend DB:** Postgres
- **Training Jobs:** Submitted to Ray Cluster
- **Runtime Environment:** PyTorch 2.2.0 with CUDA 12.1

## Serving Architecture & API

### FastAPI Endpoints
```python
POST /summarize
    ↳ request body: { "text": "<raw transcript or paragraph>" }
    ↳ response:     { "summary": "<150‑token abstractive summary>" }
```

### Model Stack
- BartTokenizerFast + BartForConditionalGeneration (base weights in `./models/base`)
- LoRA adapters (rank 64) loaded via peft.PeftModel from `./models`
- GPU inference with CPU fallback

### Deployment
- Served by Uvicorn: `uvicorn main:app --host 0.0.0.0 --port 8000`
- Nginx sidecar for TLS termination and reverse proxy
- HorizontalPodAutoscaler for GPU utilization-based scaling

### Planned Extensions
- `/qa` – Phi‑3.5 Mini Q‑A endpoint (post‑MVP)

## Monitoring & Dashboards

- **Prometheus:** Configuration in `Docker/prometheus.yml`
- **Grafana:** Panels in `Docker/grafana.yml`
- **Retraining Trigger:** Fires on data drift or accuracy drop
- **Automated Promotion:** staging → canary → production

## CI/CD & Dev Workflow

| Stage | Tool | Trigger |
|-------|------|---------|
| Model Training | AirflowDockerOperator(log_model) | DAG Scheduled daily |
| Image build| AirflowDockerOperator(build_container) | On log_model success |
| Staging Deploy |  AirflowDockerOperator(build_container) | On both build_container and offline_eval success |
| Canary release| AirflowDockerOperator(deploy_canary) | On load_test success |
| Production deploy| AirflowDockerOperator(deploy_prod) | On monitor_performance success |


### Airflow Pipeline
- DAG: `airflow/dags/audio_ml_pipeline.py`
- ONNX helpers: `airflow/dags/helpers/onnx_utils.py`
- Export pipeline: `/mnt/block/model1-artifacts/model.pth` → ONNX models

## Running on Chameleon

### Service Endpoints (Floating IP: 129.114.25.36)
- Airflow UI: [http://129.114.25.36:8080](http://129.114.25.36:8080)
- Jupyter Notebook: [http://129.114.25.36:8888](http://129.114.25.36:8888)
- MLflow Tracking: [http://129.114.25.36:8000](http://129.114.25.36:8000)
- FastAPI Backend: [http://129.114.25.36:3500](http://129.114.25.36:3500)
- Streamlit Frontend: [http://129.114.25.36:5000](http://129.114.25.36:5000)
- Grafana Dashboards: [http://129.114.25.36:3000](http://129.114.25.36:3000)

### Deployment Steps
```bash
# Deploy object‑store
helm repo add minio https://charts.min.io/
helm install minio minio/minio -f Docker/docker-object-store.yaml

# Build & push images
docker compose build
docker compose push  # requires registry creds

# Deploy micro‑services
kubectl apply -f backend/deployment/namespaces.yaml
kubectl apply -f backend/deployment/fastapi_gpu.yaml

# For Bringing up Airflow (service Orchestration )
cd /mnt/block/airflow
docker compose -f docker-compose.yaml up -d --build

# Tear down
chameleon delete --cluster transcept
```

## Contributors

| Name | Role |
|------|------|
| Aishwarya | Model Training & Experimentation |
| Bharath | Serving & Monitoring |
| Anushka | Data Pipeline & Online Eval |
| Kathan | CI/CD & Infrastructure |

