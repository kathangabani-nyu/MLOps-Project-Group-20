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
| `infrastructure/` | Chameleon launch scripts – launch_k8s_cluster.py |
| `train/` | Ray + PyLightning LoRA fine‑tunes – train_model2_ray.py |
| `docker-compose.yml` | One‑shot local deployment |

## Data Pipeline (ETL → Feature Store)

1. **Raw Lecture Audio & Video** → stored in the MinIO bucket `raw-inputs` (mounted at `/mnt/object`)
2. **Transcription** – Frontend/app.py wraps Whisper‑large‑v3. Word‑error‑rate (WER) stats logged to Prometheus
3. **Cleaning & Segmentation** – Storage/create_server.ipynb removes stopwords, normalises Unicode, and chunks text into ≤512 token windows
4. **Dataset Prep** – Data Jason files aligns CNN/DailyMail reference summaries and labels each chunk

Artifacts are versioned in the MinIO bucket `preprocessed`, path‑convention `<lecture_id>/<stage>.parquet`

**Data Dashboard:** Interactive Streamlit-based Dashboard that reads directly from block storage on kvm@tacc (mounted at `/mnt/block/MLOps-Project-Group-20/data`) providing rapid, visual feedback on raw train/validation/test JSONL datasets.

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
- `/transcribe` – Whisper ASR endpoint (work‑in‑progress)
- `/qa` – Phi‑3.5 Mini Q‑A endpoint (post‑MVP)

## Monitoring & Dashboards

- **Prometheus:** Configuration in `Docker/prometheus.yml`
- **Grafana:** Panels in `Docker/grafana.yml`
- **Retraining Trigger:** Fires on data drift or accuracy drop
- **Automated Promotion:** staging → canary → production

## CI/CD & Dev Workflow

| Stage | Tool | Trigger |
|-------|------|---------|
| Build | GitHub Actions | PR push |
| Image Publish | Docker Hub | Build success |
| Deploy | Argo CD | Image tag main-* |
| Canary + Rollback | Argo metrics | ROUGE‑L regression |

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
kubectl apply -k Monitoring/kustomize/

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

## Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->



- **Developed by:** [More Information Needed]
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** [More Information Needed]
- **Language(s) (NLP):** [More Information Needed]
- **License:** [More Information Needed]
- **Finetuned from model [optional]:** [More Information Needed]

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

[More Information Needed]

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]
### Framework versions

- PEFT 0.15.2
