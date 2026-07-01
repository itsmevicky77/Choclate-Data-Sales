# ==========================================
# STEP 12 : SAVE MODEL
# ==========================================

import os
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

print("="*60)
print("STEP 12 : SAVE MODEL")
print("="*60)

os.makedirs("models", exist_ok=True)

print("Models folder created successfully.")

train_df = pd.read_csv("Train_Data.csv")

X_train = train_df.drop("Amount", axis=1)
y_train = train_df["Amount"]

print("\nTraining Dataset Loaded")
print(X_train.shape)

final_model = RandomForestRegressor(

    n_estimators=200,

    max_depth=20,

    min_samples_split=5,

    min_samples_leaf=1,

    max_features="sqrt",

    random_state=42,

    n_jobs=-1

)

final_model.fit(X_train, y_train)

print("\nFinal Model Trained Successfully")

joblib.dump(
    final_model,
    "models/best_model.pkl"
)

print("Model Saved")

feature_columns = list(X_train.columns)

joblib.dump(
    feature_columns,
    "models/feature_columns.pkl"
)

print("Feature Columns Saved")

metadata = {

    "Model":

        "Random Forest Regressor",

    "Target":

        "Amount",

    "Training Rows":

        len(train_df),

    "Features":

        feature_columns,

    "R2 Score":

        0.9051,

    "MAE":

        37.52,

    "RMSE":

        113.87

}

with open(
    "models/model_metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

print("Metadata Saved")

print("\nSaved Files")

print("-"*40)

for file in os.listdir("models"):

    print(file)

    print("\n" + "="*60)

print("MODEL SAVED SUCCESSFULLY")

print("="*60)

print("Location : models/")

print("\nFiles Created")

print("best_model.pkl")

print("feature_columns.pkl")

print("model_metadata.json")

print("\nSTEP 12 COMPLETED SUCCESSFULLY")

