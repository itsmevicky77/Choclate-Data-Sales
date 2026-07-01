# ==========================================
# STEP 15 : MLFLOW MODEL TRACKING
# ==========================================

import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import json

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("STEP 15 : MLFLOW MODEL TRACKING")
print("=" * 60)

# ------------------------------------------
# Load Data
# ------------------------------------------

train_df = pd.read_csv("Train_Data.csv")
test_df = pd.read_csv("Test_Data.csv")

X_train = train_df.drop("Amount", axis=1)
y_train = train_df["Amount"]

X_test = test_df.drop("Amount", axis=1)
y_test = test_df["Amount"]

# ------------------------------------------
# Load Saved Model
# ------------------------------------------

model = joblib.load("models/best_model.pkl")

# ------------------------------------------
# Prediction
# ------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

# ------------------------------------------
# MLflow Setup
# ------------------------------------------

mlflow.set_experiment("Chocolate Sales Prediction")

with mlflow.start_run(run_name="Random Forest Final Model"):

    # Parameters
    mlflow.log_param("Algorithm", "Random Forest")
    mlflow.log_param("Training Rows", len(train_df))
    mlflow.log_param("Features", X_train.shape[1])

    # Metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Metadata
    mlflow.log_artifact("models/model_metadata.json")

    # Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="RandomForestModel",
        registered_model_name="ChocolateSalesRandomForest"
    )

print("\nModel Logged Successfully!")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

print("\nSTEP 15 COMPLETED SUCCESSFULLY")