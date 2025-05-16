# ⚙️ API Specification

## Endpoint: `/predict`
### Method: POST

### Request Body
```json
{
  "age": 56,
  "num_prior_admissions": 3,
  ...
}
```

### Response
```json
{
  "readmission_risk": 0.82,
  "prediction": "High Risk"
}
```

## Framework
- Built with Flask
- Dockerized for deployment
