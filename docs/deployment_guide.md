# 🚀 Deployment Guide

## Manual Steps
1. SSH into EC2
2. Clone repo and navigate to project root
3. Run Docker build and container:
```bash
docker build -t medicarenet-api .
docker run -d -p 80:5000 medicarenet-api
```

## CI/CD
Handled by GitHub Actions (`.github/workflows/deploy.yml`)
