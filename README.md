# MediCareNet 🩺 – Predictive Healthcare Insights with ML & Flask API

MediCareNet is an end-to-end machine learning pipeline designed to predict hospital readmissions based on patient EHR data. It includes preprocessing, model training (with Truncated SVD and XGBoost), and a Flask-based API for real-time predictions, fully containerized using Docker.


## 📁 Project Structure

```plaintext
MediCareNet/
├── app.py                        # Flask API for inference (serves /predict)
├── Dockerfile                    # Defines the container image
├── docker-compose.yml            # Docker Compose to run the service
├── requirements.txt              # Python dependencies

├── test/                         # Testing inputs and utilities
│   └── sample_input_components.json  # Sample input (50D SVD vector) for testing /predict

├── data/                         # Dataset files
│   ├── cleaned_data.csv          # Raw cleaned EHR dataset
│   └── feature_engineered_data.csv  # Output from preprocessing

├── src/                          # Source code
│   ├── preprocess.py             # Data cleaning & feature engineering script
│   ├── train.py                  # Model training pipeline
│   └── models/                   # Folder for trained models
│       └── XGBoost_model.pkl     # Trained XGBoost model (SVD-reduced input)
```

## 🚀 Getting Started

### 1. 📦 Clone the Repository

```bash
git clone https://github.com/chaitrasutari/MediCareNet.git
cd MediCareNet
```


### 2. 🐳 Run the Project via Docker

If You ONLY Want to Run Predictions via Docker

Make sure Docker Desktop is installed and running.

#### Build and start the Flask API:
```bash
docker compose up --build
```

You should see:
```bash
* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5000
```

## 📬 API Endpoints

### 🔹 `/predict` [POST]
Accepts a 50-dimensional vector (output of Truncated SVD) and returns a binary prediction (0 or 1).

#### Sample Request
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @test/sample_input_components.json
```

#### Sample Response
```json
{"prediction": 0}
```


## 🧠 Do You Need to Retrain the Model?

### Full pipeline:

#### Run preprocessing:
```bash
python src/preprocess.py --input data/processed/cleaned_data.csv --output data/processed/feature_engineered_data.csv
```

#### Train the model:
```bash
python src/models/train.py --input data/processed/feature_engineered_data.csv --model_out src/models/XGBoost_model.pkl
```

#### Then build and run the API:
```bash
docker compose up --build
```

---

| Scenario                      | Action                                           |
|------------------------------|--------------------------------------------------|
| Use the pre-trained model | Just run `docker compose up --build`            |
| Retrain the model         | Run `preprocess.py` → `train.py` → then Docker  |


## ⚙️ MLOps Overview

This project integrates several essential MLOps principles to ensure reproducibility, modularity, scalability, and deployment-readiness.



### MLOps Practices

| Category            | Tools/Approach              | Description |
|---------------------|-----------------------------|-------------|
| **Experiment Tracking** | `MLflow` (`train.py`)     | Logs models, hyperparameters, and metrics like F1 and ROC-AUC. Each model run is traceable and reproducible. |
| **Reproducibility**     | `joblib`, `train.py` CLI args | Trained models are saved as `.pkl` files; training parameters are configurable via command-line arguments. |
| **Modular Code**        | `preprocess.py`, `train.py`, `app.py` | Clear separation between preprocessing, model training, and API inference logic. |
| **Model Packaging**     | `Flask + Docker`          | The model is served using a lightweight Flask API, wrapped in a Docker container for consistent deployment. |
| **Local Deployment**    | `docker-compose`          | Spin up the inference server locally with one command using Docker Compose. |


