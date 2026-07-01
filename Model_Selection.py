# ==========================================
# STEP 11 : MODEL SELECTION
# ==========================================

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("=" * 60)
print("STEP 11 : MODEL SELECTION")
print("=" * 60)

train_df = pd.read_csv("Train_Data.csv")
test_df = pd.read_csv("Test_Data.csv")

X_train = train_df.drop("Amount", axis=1)
y_train = train_df["Amount"]

X_test = test_df.drop("Amount", axis=1)
y_test = test_df["Amount"]

print("\nDatasets Loaded Successfully")

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    ),

    "Tuned Random Forest": RandomForestRegressor(

        random_state=42,

        n_estimators=200,

        max_depth=20,

        min_samples_split=5,

        min_samples_leaf=1,

        max_features="sqrt",

        n_jobs=-1
    )

}

results = []

best_model = None
best_r2 = -1

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    results.append({

        "Model": name,

        "MAE": round(mae,2),

        "RMSE": round(rmse,2),

        "R2 Score": round(r2,4)

    })

    if r2 > best_r2:

        best_r2 = r2

        best_model = model

        best_model_name = name

        results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Model : {best_model_name}")
print(f"R² Score : {best_r2:.4f}")

results_df.to_csv(
    "Model_Comparison_Report.csv",
    index=False
)

print("\nModel Comparison Report Saved Successfully!")

print("\n" + "=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)

print("Selected Model :", best_model_name)

print("Reason : Highest R² Score")

print(f"Final R² : {best_r2:.4f}")

print("\nSTEP 11 COMPLETED SUCCESSFULLY")

