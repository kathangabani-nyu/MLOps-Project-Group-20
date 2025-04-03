# Transcept
#LoRA Fine-tuned Summarization and QA for Lecture Transcripts

## Goal

We propose a machine learning system to enhance lecture accessibility for university students by automating the generation of summaries and enabling question answering from lecture content. The system integrates seamlessly into existing LMS (Learning Management System) platforms (Canvas, Moodle, etc.).

---

## Value Proposition

**Non-ML status quo:**  
Students must watch lengthy recordings or rely on manual note-taking. Most academic platforms do not offer real-time summarization or question answering.

**ML solution — EduPulse:**  
- Transcribes lectures using Whisper ASR  
- Summarizes content with a LoRA fine-tuned Mistral-7B Instruct model  
- Answers student questions using a LoRA fine-tuned Phi-3.5 Mini model  

**Business metric:**  
Increased user engagement and reduced lecture playback time.

---

## Contributors

All Members: Ideation, integration, infra, pipelines  
- **Aishwarya**: Model training (Units 4 & 5)  
- **Anushka**: Model serving and monitoring (Units 6 & 7)  
- **Bharat**: Data pipeline (Unit 8)  
- **Kathan**: Continuous Integration/Delivery (Unit 3)

---

## System Diagram

1. Lecture video/audio (input)  
2. Whisper ASR (audio → text)  
3. Preprocessed transcript  
4. Mistral-7B Instruct (summary generation)  
5. Phi-3.5 Instruct (question answering)  
6. API exposed via FastAPI or TGI  
7. Load testing & monitoring with Prometheus/Grafana  
8. CI/CD via Argo Workflows  
9. Experiment tracking using MLflow and Ray clusters

---

## External Materials Summary

- **Whisper ASR**: OpenAI model trained on multilingual speech (MIT License)  
- **Mistral-7B Instruct**: Open-weight decoder model (Apache 2.0)  
- **Phi-3.5 Instruct**: Microsoft foundation model (MIT License)  
- **SQuAD v2 Dataset**: Stanford QA dataset (CC BY-SA 4.0)  
- **Lecture Audio**: Simulated/open lectures (Fair use for academic purposes)  
- **TED-LIUMv2 (corpus)**

---

## Infrastructure Requirements

- `m1.medium` VMs: 2 for full project duration (API hosting and orchestration)  
- `gpu_a100`: Twice weekly, 4-hour blocks (LoRA fine-tuning)  
- Floating IP: Public access to inference API  
- Volume Storage: 50GB persistent (data/models/logs)  
- Object Storage (e.g., S3 or Swift): On-demand (raw audio/video inputs, model checkpoints, outputs)  
- MLflow Server: Full project duration (training run tracking, model performance, and metrics)

---

## Design Plan (by Component)

### Model Training – Aishwarya
- Strategy: `trl.SFTTrainer` with LoRA adapters (`peft`). Format: `instruction + input → output`  
- Infra: Chameleon cloud, Ray, MLflow. FlashAttention preferred.  
- Difficulty: Efficient large-model fine-tuning (Units 4 & 5)

### Model Serving & Monitoring – Anushka
- Strategy: FastAPI + vLLM/TGI, Prometheus-based metrics  
- Optimization: 4-bit vs. full precision inference  
- Difficulty: Load tests + canary deployments (Units 6 & 7)

### Data Pipeline – Bharath
- ETL: Whisper → transcript → clean → segment → summary/QA  
- Simulated ingestion via scripts/Kafka  
- Storage: Persistent volumes  
- Difficulty: Status/failure dashboards (Unit 8)

### CI/CD – Kathan
- GitHub Actions → Docker → Argo  
- Retraining triggers: schedule or drift detection  
- Difficulty: Canary testing + rollback (Unit 3)

---

## Difficulty Points

- ✅ Unit 1: Multi-model pipeline (ASR → Summarizer → QA)  
- ✅ Unit 7: Degradation detection triggering retraining
