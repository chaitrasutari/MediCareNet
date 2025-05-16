# 🐳 Docker Guide

## Dockerfile
Includes:
- Python 3.10 base
- Flask + scikit-learn + dependencies
- Copying model and API code

## Build & Run
```bash
docker build -t medicarenet-api .
docker run -p 5000:5000 medicarenet-api
```

## Docker Compose (optional)
Add `docker-compose.yml` for running MLflow + API
