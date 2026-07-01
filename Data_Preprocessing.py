# ==========================================
# STEP 7 : DATA PREPROCESSING
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("=" * 60)
print("STEP 7 : DATA PREPROCESSING")
print("=" * 60)

# Load Selected Features Dataset

df = pd.read_csv("Chocolate_Sales_Selected_Features.csv")

print("\nDataset Loaded Successfully")
print("Shape :", df.shape)

X = df.drop("Amount", axis=1)

y = df["Amount"]

print("\nFeatures Shape :", X.shape)
print("Target Shape   :", y.shape)

numerical_features = [

    "Boxes_Shipped",

    "Price_per_Box",

    "Marketing_Spend",

    "Discount_Pct"

]

print("\nNumerical Features")

print(numerical_features)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("\nTraining Shape")

print(X_train.shape)

print("\nTesting Shape")

print(X_test.shape)

print("\nScaling Numerical Features...")

scaler = StandardScaler()

X_train[numerical_features] = scaler.fit_transform(

    X_train[numerical_features]

)

X_test[numerical_features] = scaler.transform(

    X_test[numerical_features]

)



import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaling Completed and Scaler Saved Successfully")

print("\nScaled Training Data")

print(

    X_train[numerical_features].describe()

)

train_df = X_train.copy()

train_df["Amount"] = y_train

test_df = X_test.copy()

test_df["Amount"] = y_test

train_df.to_csv(

    "Train_Data.csv",

    index=False

)

test_df.to_csv(

    "Test_Data.csv",

    index=False

)

print("\nProcessed Dataset Saved Successfully!")

print("\n" + "=" * 60)

print("FINAL DATASET")

print("=" * 60)

print("\nTraining Dataset")

print(train_df.shape)

print("\nTesting Dataset")

print(test_df.shape)

print("\nSTEP 7 COMPLETED SUCCESSFULLY")

