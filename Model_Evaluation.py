# ==========================================
# STEP 9 : MODEL EVALUATION
# ==========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width",150)

print("="*60)
print("STEP 9 : MODEL EVALUATION")
print("="*60)

train_df = pd.read_csv("Train_Data.csv")
test_df = pd.read_csv("Test_Data.csv")

X_train = train_df.drop("Amount", axis=1)
y_train = train_df["Amount"]

X_test = test_df.drop("Amount", axis=1)
y_test = test_df["Amount"]

print("\nDatasets Loaded Successfully")

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

predictions = rf.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-"*40)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

residuals = y_test - predictions

import os

os.makedirs("evaluation_images", exist_ok=True)

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.4
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.xlabel("Actual Amount")

plt.ylabel("Predicted Amount")

plt.title("Actual vs Predicted")

plt.tight_layout()

plt.savefig("evaluation_images/actual_vs_predicted.png")

plt.show()

plt.figure(figsize=(8,6))

plt.scatter(
    predictions,
    residuals,
    alpha=0.4
)

plt.axhline(
    y=0,
    color="red"
)

plt.xlabel("Predicted Amount")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig("evaluation_images/residual_plot.png")

plt.show()

plt.figure(figsize=(8,6))

plt.hist(
    residuals,
    bins=50
)

plt.xlabel("Residual")

plt.ylabel("Frequency")

plt.title("Residual Distribution")

plt.tight_layout()

plt.savefig("evaluation_images/residual_distribution.png")

plt.show()

errors = abs(residuals)

plt.figure(figsize=(8,6))

plt.hist(
    errors,
    bins=50
)

plt.xlabel("Absolute Error")

plt.ylabel("Frequency")

plt.title("Prediction Error Distribution")

plt.tight_layout()

plt.savefig("evaluation_images/prediction_error_distribution.png")

plt.show()

importance = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
)

importance = importance.sort_values()

plt.figure(figsize=(8,6))

plt.barh(
    importance.index,
    importance.values
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("evaluation_images/feature_importance.png")

plt.show()

print("\n" + "="*60)
print("MODEL EVALUATION SUMMARY")
print("="*60)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

print("\nEvaluation graphs saved in:")
print("evaluation_images/")