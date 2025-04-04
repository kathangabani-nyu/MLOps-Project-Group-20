# Transcept  
LoRA Fine-tuned Summarization and QA for Lecture Transcripts

## Value Proposition

**Current status quo:**  
Students in universities often struggle to keep up with lecture recordings, which are long and lack intelligent navigation. LMS platforms like Canvas or Moodle do not support real-time summarization or question answering.

**ML-based system:**  
We propose **Transcept**, a machine learning system integrated into LMS platforms that:
- Transcribes lecture recordings using Whisper ASR.
- Generates concise summaries using a **LoRA fine-tuned Mistral-7B Instruct** model.
- Enables students to ask questions based on the lecture using a **LoRA fine-tuned Phi-3.5 Mini** model.

**Business metric:**  
We aim to **reduce lecture playback time** and **increase student engagement** with recorded content.

---

## Contributors

| Name        | Responsible for                      |
|-------------|--------------------------------------|
| All members | Ideation, integration, pipelines     |
| Aishwarya   | Model Training (Units 4 & 5)          |
| Bharath     | Model Serving & Monitoring (Units 6 & 7) |
| Anushka     | Data Pipeline (Unit 8)               |
| Kathan      | Continuous X Pipeline (Unit 3)       |

---

## System Diagram

![Blank diagram (3)](https://github.com/user-attachments/assets/7a55c086-b0df-45bd-8a0b-ab84f2a85003)

---

## Summary of Outside Materials

| Name              | How it was created                                                                 | Conditions of use                          |
|-------------------|--------------------------------------------------------------------------------------|---------------------------------------------|
| Whisper ASR       | OpenAI multilingual ASR model                                                       | MIT License                                 |
| TED-LIUM v2       | Speech corpus from TED talks, used for ASR finetuning                              | Academic use; free for research             |
| Mistral-7B Instruct | Open-weight LLM decoder (Mistral.ai, fine-tuned for instruction following)       | Apache 2.0                                  |
| Phi-3.5 Mini Instruct | Microsoft foundation model for QA and reasoning                                | MIT License                                 |
| SQuAD v2          | Stanford QA dataset with unanswerable questions                                     | CC BY-SA 4.0                                |
| Lecture Audio     | Simulated/open lectures; NYU class material (educational fair use)                  | Fair use for academic purposes              |

---

## Summary of Infrastructure Requirements

| Requirement       | How many/when                        | Justification                                               |
|------------------|--------------------------------------|-------------------------------------------------------------|
| `m1.medium` VMs   | 2 for full project duration          | API hosting, orchestration, data pipeline                   |
| `gpu_a100`        | 2x/week, 4 hours                     | LoRA finetuning of large models                             |
| Floating IP       | 1 for full duration                  | Public access for API/monitoring                           |
| Volume Storage    | 50GB persistent                      | Logs, preprocessed transcripts, checkpoints, etc.           |
| Object Storage    | On-demand (S3 or Swift)              | Raw audio/video inputs, finetuned models, summaries         |
| MLflow Server     | Full duration                        | Track training runs, evaluation, performance metrics        |

---

## Detailed Design Plan

### Model Training and Training Platforms (Aishwarya)

- **Strategy**: LoRA finetuning with `trl.SFTTrainer` using `peft`, in `instruction + input → output` format.
- **Infra**: Runs on Chameleon cloud with Ray cluster and MLflow tracking.
- **Techniques**: FlashAttention, bfloat16 precision, scheduled retraining.
- **Difficulty points**:
  - ✅ *Distributed Training via Ray Train*  
  - ✅ *Training time vs GPU scaling (1 vs multi-GPU)*  
- **Evaluation Plan**: After each run, models are automatically evaluated and logged to MLflow. Models that exceed threshold are registered.

### Model Serving and Monitoring Platforms (Bharath)

- **Strategy**: FastAPI backend with TGI/vLLM for hosting models.
- **Optimizations**: Compare 4-bit quantization vs full precision.
- **Monitoring**: Prometheus & Grafana dashboards for latency, throughput, error rate.
- **Difficulty points**:
  - ✅ *Serve both quantized and full-precision variants*  
  - ✅ *Trigger re-training on model degradation*  

### Data Pipeline (Anushka)

- **ETL Steps**:
  - Whisper transcription → clean text → segmentation → inputs to summarizer & QA
- **Storage**: Persistent volumes for intermediate artifacts; object store for raw inputs.
- **Streaming**: Simulated Kafka-based online data stream for inference QA testing.
- **Difficulty point**:
  - ✅ *Interactive data dashboard* for transcript health and error stats

### Continuous X (Kathan)

- **CI/CD Stack**: GitHub Actions → Docker → Argo Workflows
- **Triggers**: Model drift, schedule, manual
- **Staging/Canary/Prod**: Models are promoted via pipelines with validation at each stage
- **Difficulty point**:
  - ✅ *Canary deployments + automated rollback*
