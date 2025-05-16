# 📡 MLflow Remote Setup

## Backend Store
Use SQLite or PostgreSQL mounted volume

## Artifact Store
Configure with:
```bash
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
mlflow server   --backend-store-uri sqlite:///mlflow.db   --default-artifact-root s3://your-bucket/mlflow-artifacts
```
