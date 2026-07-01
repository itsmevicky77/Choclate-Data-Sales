# ==========================================
# STEP 6 : FEATURE SELECTION
# Chocolate Sales Prediction
# ==========================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

# Display Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("=" * 60)
print("STEP 6 : FEATURE SELECTION")
print("=" * 60)

# ==========================================
# Load Feature Engineered Dataset
# ==========================================

df = pd.read_csv("Chocolate_Sales_Feature_Engineered.csv")

print("\nDataset Loaded Successfully")
print("Shape :", df.shape)

# ==========================================
# Remove Unnecessary Columns
# ==========================================

print("\nRemoving Unnecessary Columns...")

columns_to_drop = [

    "Order_ID",              # Unique Identifier

    "Order_Date",            # Already engineered

    "Month_Name",            # Duplicate information

    "Revenue_Per_Box",       # Target Leakage

    "Marketing_Efficiency"   # Target Leakage

]

df.drop(columns=columns_to_drop, inplace=True)

print("Shape After Dropping Columns :", df.shape)

# ==========================================
# Encode Categorical Variables
# ==========================================

print("\nEncoding Categorical Variables...")

import joblib
import os

os.makedirs("models", exist_ok=True)

label_encoders = {}

categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    label_encoders[column] = encoder

joblib.dump(
    label_encoders,
    "models/label_encoders.pkl"
)

print("Label Encoders Saved Successfully")

print("Encoding Completed")

# ==========================================
# Separate Features and Target
# ==========================================

X = df.drop("Amount", axis=1)

y = df["Amount"]

print("\nFeatures Shape :", X.shape)
print("Target Shape   :", y.shape)

# ==========================================
# Correlation Analysis
# ==========================================

print("\n" + "=" * 60)
print("CORRELATION WITH TARGET")
print("=" * 60)

correlation = df.corr(numeric_only=True)

print(
    correlation["Amount"]
    .sort_values(ascending=False)
)

# ==========================================
# Mutual Information
# ==========================================

print("\n" + "=" * 60)
print("MUTUAL INFORMATION")
print("=" * 60)

mi_scores = mutual_info_regression(
    X,
    y,
    random_state=42
)

mi = pd.Series(
    mi_scores,
    index=X.columns
).sort_values(ascending=False)

print(mi)

# ==========================================
# Random Forest Feature Importance
# ==========================================

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance)

# ==========================================
# Final Feature Selection
# ==========================================

print("\n" + "=" * 60)
print("FINAL SELECTED FEATURES")
print("=" * 60)

# Selected using:
# ✔ Business Knowledge
# ✔ Correlation
# ✔ Mutual Information
# ✔ Random Forest Importance

selected_features = [

    "Boxes_Shipped",

    "Price_per_Box",

    "Marketing_Spend",

    "Discount_Pct",

    "Product",

    "Country",

    "Channel",

    "Month",

    "Quarter"

]

print(selected_features)

# ==========================================
# Create Final Dataset
# ==========================================

selected_df = df[selected_features + ["Amount"]]

print("\nFinal Dataset Shape")
print(selected_df.shape)

# ==========================================
# Save Dataset
# ==========================================

selected_df.to_csv(
    "Chocolate_Sales_Selected_Features.csv",
    index=False
)

print("\nSelected Features Dataset Saved Successfully!")

print("\n" + "=" * 60)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 60)