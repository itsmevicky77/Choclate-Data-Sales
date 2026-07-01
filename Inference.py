# ==========================================
# STEP 13 : INFERENCE PIPELINE
# ==========================================

import joblib
import pandas as pd

print("=" * 60)
print("STEP 13 : INFERENCE PIPELINE")
print("=" * 60)

# ==========================================
# Load Saved Files
# ==========================================

print("\nLoading Saved Model...")

model = joblib.load("models/best_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
encoders = joblib.load("models/label_encoders.pkl")
scaler = joblib.load("models/scaler.pkl")

print("Model Loaded Successfully")

# ==========================================
# New User Input
# (Use actual values instead of encoded numbers)
# ==========================================

sample_data = {

    "Boxes_Shipped": [150],

    "Price_per_Box": [3.75],

    "Marketing_Spend": [120],

    "Discount_Pct": [10],

    "Product": ["70% Dark Bar"],

    "Country": ["Australia"],

    "Channel": ["Retail"],

    "Month": [7],

    "Quarter": [3]

}

# ==========================================
# Convert to DataFrame
# ==========================================

input_df = pd.DataFrame(sample_data)

print("\nOriginal Input")
print(input_df)

# ==========================================
# Encode Categorical Columns
# ==========================================

input_df["Product"] = encoders["Product"].transform(input_df["Product"])

input_df["Country"] = encoders["Country"].transform(input_df["Country"])

input_df["Channel"] = encoders["Channel"].transform(input_df["Channel"])

# ==========================================
# Scale Numerical Columns
# ==========================================

numeric_columns = [

    "Boxes_Shipped",

    "Price_per_Box",

    "Marketing_Spend",

    "Discount_Pct"

]

input_df[numeric_columns] = scaler.transform(
    input_df[numeric_columns]
)

# ==========================================
# Arrange Columns
# ==========================================

input_df = input_df[feature_columns]

print("\nProcessed Input")
print(input_df)

# ==========================================
# Predict
# ==========================================

prediction = model.predict(input_df)

print("\nPredicted Chocolate Sales Amount")

print(f"${prediction[0]:,.2f}")

print("\n" + "=" * 60)
print("PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 60)