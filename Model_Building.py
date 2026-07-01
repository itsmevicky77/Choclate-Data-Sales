# ==========================================
# STEP 8 : MODEL BUILDING
# Chocolate Sales Prediction
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Display Settings
# ==========================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("=" * 60)
print("STEP 8 : MODEL BUILDING")
print("=" * 60)

# ==========================================
# Load Train & Test Data
# ==========================================

train_df = pd.read_csv("Train_Data.csv")
test_df = pd.read_csv("Test_Data.csv")

print("\nTraining Dataset :", train_df.shape)
print("Testing Dataset  :", test_df.shape)

# ==========================================
# Split Features & Target
# ==========================================

X_train = train_df.drop("Amount", axis=1)
y_train = train_df["Amount"]

X_test = test_df.drop("Amount", axis=1)
y_test = test_df["Amount"]

print("\nTraining Features :", X_train.shape)
print("Testing Features  :", X_test.shape)

# ==========================================
# Models
# ==========================================

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(

        n_estimators=100,

        max_depth=20,

        min_samples_split=5,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1

    )

}

results = []

trained_models = {}

# ==========================================
# Train Models
# ==========================================

for name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Training {name}")
    print("=" * 50)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    trained_models[name] = model

    results.append({

        "Model": name,

        "MAE": round(mae, 2),

        "RMSE": round(rmse, 2),

        "R2 Score": round(r2, 4)

    })

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

# ==========================================
# Model Comparison
# ==========================================

results_df = pd.DataFrame(results)

results_df.sort_values(

    by="R2 Score",

    ascending=False,

    inplace=True

)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

# ==========================================
# Best Model
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Selected Model : {best_model_name}")

print(results_df.iloc[0])

# ==========================================
# Save Comparison Report
# ==========================================

results_df.to_csv(

    "Model_Comparison_Report.csv",

    index=False

)

print("\nModel Comparison Report Saved Successfully!")

# ==========================================
# Save Best Model
# ==========================================

joblib.dump(

    best_model,

    "Random_Forest_Model.pkl"

)

print("\nBest Model Saved Successfully!")

# ==========================================
# Completed
# ==========================================

print("\n" + "=" * 60)
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 60)