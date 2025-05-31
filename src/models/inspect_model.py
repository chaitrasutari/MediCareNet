import joblib

# Load the trained model
model = joblib.load("src\models\XGBoost_model.pkl")

# Get feature names (if available)
if hasattr(model, "feature_names_in_"):
    print("Model expects these features:\n")
    for feature in model.feature_names_in_:
        print(f"- {feature}")
else:
    print("❌ This model does not have `feature_names_in_` attribute.")
