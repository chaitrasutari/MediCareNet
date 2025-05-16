# 🔄 DVC Pipeline

## Stages
- Data ingestion
- Preprocessing
- Model training
- Evaluation

## Commands
```bash
dvc init
dvc add data/raw.csv
dvc run -n train_model -d src/train.py -o models/model.pkl python src/train.py
```
