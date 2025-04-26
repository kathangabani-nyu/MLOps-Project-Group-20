# MLOps Project - Group 20

## Infrastructure Components

- ArgoCD: Continuous Delivery tool (Port 30500)
- MinIO: Object Storage (Port 30900)
- MLflow: ML Experiment Tracking (Port 30500)

## Getting Started

1. Access ArgoCD UI:
   - URL: http://<node-ip>:30500
   - Username: admin
   - Password: (use the one generated earlier)

2. Access MinIO:
   - URL: http://<node-ip>:30900
   - Access Key: minio
   - Secret Key: minio123

3. Access MLflow:
   - URL: http://<node-ip>:30500

## Development

1. Clone the repository
2. Create a new branch for your feature
3. Make changes and commit
4. Create a pull request

## Project Structure

```
.
├── infrastructure/
│   ├── argocd/
│   ├── monitoring/
│   ├── mlflow/
│   └── minio/
├── src/
│   ├── transcription/
│   ├── processing/
│   ├── summary/
│   └── api/
├── pipelines/
│   └── workflows/
└── .github/
    └── workflows/
```
