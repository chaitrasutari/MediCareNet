#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import joblib


def main(data_path, model_output_path):
    # In[4]:


    # ----------------------
    # Step 1: Load Data
    # ----------------------
    # data_path = "../data/processed/feature_engineered_data.csv"  # Adjust path if needed
    df = pd.read_csv(data_path)
        

    # In[5]:


    # ----------------------
    # Step 2: Separate features and target
    # ----------------------
    target_col = "readmitted_30"
    X = df.drop(columns=[target_col])
    y = df[target_col]


    # In[6]:


    # ----------------------
    # Step 3: Identify categorical columns (object dtype)
    # ----------------------
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"Categorical columns to encode: {cat_cols}")


    # In[7]:


    # ----------------------
    # Step 4: One-hot encode categorical columns
    # ----------------------
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)


    # In[8]:


    # Clean column names (optional, recommended for XGBoost)
    def clean_column_names(columns):
        return columns.str.replace(r"[\[\]\(\)<>\-]", "_", regex=True).str.replace("__", "_")


    # In[9]:


    X_encoded.columns = clean_column_names(X_encoded.columns)


    # In[10]:

    svd = TruncatedSVD(n_components=500, random_state=42)
    svd.fit(X_encoded)

    explained_variance = svd.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    plt.plot(range(1, 501), cumulative_variance)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.show()


    # In[35]:


    # Apply TruncatedSVD for dimensionality reduction
    n_components = 50  # Adjust based on memory/performance trade-off
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    X_reduced = svd.fit_transform(X_encoded)


    # In[36]:


    print(f"Reduced feature shape: {X_reduced.shape}")


    # In[37]:


    # ----------------------
    # Step 2: Train-Test Split
    # ----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42, stratify=y
    )


    # In[38]:


    # ----------------------
    # Step 6: Define models to train
    # ----------------------
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=42),
    }


    # In[39]:


    # ----------------------
    # Step 7: Cross-validation results for comparison
    # ----------------------
    print("Cross-validation F1 scores:")

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
        print(f" - {name}: mean={scores.mean():.4f}, std={scores.std():.4f}")


    # In[40]:


    mlflow.set_experiment("mediCareNet-readmission")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Metrics
            f1 = f1_score(y_test, preds)
            roc_auc = roc_auc_score(y_test, proba) if proba is not None else None
            
            # Infer model signature from training data
            signature = infer_signature(X_train, model.predict(X_train))
            
            # Log parameters and metrics
            mlflow.log_params(model.get_params())
            mlflow.log_metric("f1_score", f1)
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)
            
            # Log model with signature and input example (first 5 rows of train)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train[:5]
            )
            
            print(f"{name} trained and logged with signature.")
            print(f"   F1 Score: {f1:.4f}", f"ROC-AUC: {roc_auc:.4f}" if roc_auc else "")


    # In[41]:

    # Models to tune
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42),
    }

    # Parameter distributions for RandomizedSearchCV
    param_distributions = {
        "LogisticRegression": {
            "C": uniform(loc=0.01, scale=10),          # Regularization strength
            "penalty": ["l2"],                         # L1 requires solver='liblinear', so keep L2 for simplicity
            "solver": ["lbfgs", "saga"],
            "class_weight": [None, "balanced"]
        },
        "RandomForest": {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 20),
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 5),
            "bootstrap": [True, False],
            "class_weight": [None, "balanced"]
        },
        "XGBoost": {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 20),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "gamma": uniform(0, 5),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(0, 1),
        }
    }

    # Number of parameter settings to sample
    n_iter_search = 20

    # Perform tuning for each model
    for name, model in models.items():
        print(f"Hyperparameter tuning for {name}...")
        param_dist = param_distributions[name]
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring="f1",
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best params for {name}: {random_search.best_params_}")
        print(f"Best CV F1 score for {name}: {random_search.best_score_:.4f}")
        
        # Update the model in dict with best estimator
        models[name] = random_search.best_estimator_


    # In[42]:

    print("Test set performance of tuned models:")

    for name, model in models.items():
        # Predict on test set
        preds = model.predict(X_test)
        
        # Accuracy
        acc = accuracy_score(y_test, preds)
        
        # F1 Score
        f1 = f1_score(y_test, preds)
        
        # ROC-AUC (if probability available)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, proba)
        else:
            roc_auc = None
        
        print(f"{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"  ROC-AUC: {roc_auc:.4f}")
        print()


    # In[ ]:


    # Calculate scale_pos_weight
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count

    # Create XGBoost with scale_pos_weight
    xgb_model = xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )

    # Fit and evaluate as before
    xgb_model.fit(X_train, y_train)
    preds = xgb_model.predict(X_test)
    proba = xgb_model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, proba)

    print(f"F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")


    # In[47]:

    best_threshold = 0.5
    best_f1 = 0

    thresholds = [i * 0.01 for i in range(1, 100)]  # 0.01 to 0.99

    for threshold in thresholds:
        preds = (proba >= threshold).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best threshold: {best_threshold:.2f} with F1: {best_f1:.4f}")


    # In[48]:

    f1_scores = []

    for threshold in thresholds:
        preds = (proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_test, preds))

    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Classification Threshold")
    plt.show()


    # In[55]:
    mlflow.set_tracking_uri("file:///C:/Users/HOME/Downloads/MediCareNet/mlruns")
    mlflow.set_experiment("mediCareNet-readmission")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            
            # Predict probabilities if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]
                
                # Tune threshold for best F1
                thresholds = np.linspace(0.01, 0.99, 99)
                f1_scores = []
                for t in thresholds:
                    preds = (proba >= t).astype(int)
                    f1_scores.append(f1_score(y_test, preds))
                
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx]
                best_f1 = f1_scores[best_idx]
                
                # Use best threshold for final preds
                final_preds = (proba >= best_threshold).astype(int)
                
                roc_auc = roc_auc_score(y_test, proba)
                
            else:
                # For models without predict_proba
                final_preds = model.predict(X_test)
                best_threshold = 0.5  # default
                best_f1 = f1_score(y_test, final_preds)
                roc_auc = None
            
            # Log parameters and metrics including best threshold
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.log_params(model.get_params())
            mlflow.log_metric("f1_score", best_f1)
            mlflow.log_metric("best_threshold", best_threshold)
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)
            
            # Log model with signature and example
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train[:5]
            )
            
            # Before saving model to disk
            os.makedirs("models", exist_ok=True)

            # Save model to disk
            model_filename = f"src/models/{name}_model.pkl"
            joblib.dump(model, model_filename)
            print(f"Model saved to {model_filename}")

            
            print(f"{name} trained and logged with adaptive threshold {best_threshold:.2f}")
            print(f"   Best F1 Score: {best_f1:.4f}")
            if roc_auc is not None:
                print(f"   ROC-AUC: {roc_auc:.4f}")
            print(classification_report(y_test, final_preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed CSV input")
    parser.add_argument("--model_out", required=True, help="Path to save trained model (not used directly here, but you can adapt)")
    args = parser.parse_args()
    main(args.input, args.model_out)