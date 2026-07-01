# ==========================================
# STEP 3 : DATA CLEANING
# Chocolate Sales Analysis
# ==========================================

import pandas as pd
import numpy as np

# Display Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("=" * 60)
print("STEP 3 : DATA CLEANING")
print("=" * 60)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Chocolate_Sales.csv")

print("\nDataset Loaded Successfully")
print(f"Original Shape : {df.shape}")

# ==========================================
# Missing Values Before Cleaning
# ==========================================

print("\nMissing Values Before Cleaning")
print("-" * 40)
print(df.isnull().sum())

# ==========================================
# Convert Order_Date
# ==========================================

print("\nSample Dates")
print(df["Order_Date"].head())

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

print("\nOrder_Date converted successfully")
print(df["Order_Date"].dtype)

# ==========================================
# Convert Amount
# ==========================================

print("\nSample Amount Values")
print(df["Amount"].head(10))

# Save Original Amount
df["Amount_Original"] = df["Amount"]

# Remove Dollar Symbol
df["Amount"] = (
    df["Amount"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.strip()
)

# Convert to Numeric
df["Amount"] = pd.to_numeric(
    df["Amount"],
    errors="coerce"
)


print("\nAmount datatype")
print(df["Amount"].dtype)

# ==========================================
# Verify Data Types
# ==========================================

print("\nCurrent Data Types")
print(df.dtypes)

# ==========================================
# Missing Value Analysis
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

numeric_columns = [
    "Discount_Pct",
    "Price_per_Box",
    "Marketing_Spend"
]

for col in numeric_columns:
    print(f"\n{col}")
    print("-" * 40)
    print(df[col].describe())

# ==========================================
# Check Skewness
# ==========================================

print("\n" + "=" * 60)
print("SKEWNESS")
print("=" * 60)

print(df[numeric_columns].skew())

# ==========================================
# Handle Missing Values
# ==========================================

print("\n" + "=" * 60)
print("HANDLING MISSING VALUES")
print("=" * 60)

# Drop Missing Dates
df.dropna(subset=["Order_Date"], inplace=True)

# Fill Numerical Missing Values

df["Discount_Pct"] = df["Discount_Pct"].fillna(
    df["Discount_Pct"].mean()
)

df["Price_per_Box"] = df["Price_per_Box"].fillna(
    df["Price_per_Box"].median()
)

df["Marketing_Spend"] = df["Marketing_Spend"].fillna(
    df["Marketing_Spend"].median()
)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

print("\nDataset Shape After Cleaning")
print(df.shape)

# ==========================================
# Check Invalid Amount Values
# ==========================================

print("\n" + "=" * 60)
print("INVALID AMOUNT VALUES")
print("=" * 60)

invalid_amount = df[df["Amount"].isnull()]

print(f"Number of Invalid Amount Records : {len(invalid_amount)}")

print(
    invalid_amount[
        [
            "Amount_Original",
            "Amount",
            "Product",
            "Boxes_Shipped"
        ]
    ].head(10)
)

# ==========================================
# Check Invalid Boxes
# ==========================================

print("\n" + "=" * 60)
print("INVALID BOXES SHIPPED")
print("=" * 60)

invalid_boxes = df[df["Boxes_Shipped"] < 0]

print(f"Number of Invalid Records : {len(invalid_boxes)}")

print(
    invalid_boxes[
        [
            "Order_ID",
            "Product",
            "Country",
            "Boxes_Shipped",
            "Amount"
        ]
    ].head(10)
)

print("\nMinimum Boxes Shipped :", df["Boxes_Shipped"].min())
print("Maximum Boxes Shipped :", df["Boxes_Shipped"].max())

# ==========================================
# Remove Invalid Boxes
# ==========================================

print("\nRemoving invalid Boxes_Shipped records...")

rows_before = len(df)

df = df[df["Boxes_Shipped"] >= 0].copy()

rows_after = len(df)

print(f"Rows Removed : {rows_before - rows_after}")
print(f"Current Shape : {df.shape}")

# ==========================================
# Remove Debug Column
# ==========================================

if "Amount_Original" in df.columns:
    df.drop(columns=["Amount_Original"], inplace=True)
    
# ==========================================
# Final Validation
# ==========================================

print("\n" + "=" * 60)
print("FINAL DATA VALIDATION")
print("=" * 60)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

print("\nData Types")
print(df.dtypes)

print("\nFinal Dataset Shape")
print(df.shape)

# ==========================================
# Drop Debug Column
# ==========================================

if "Amount_Original" in df.columns:
    df.drop(columns=["Amount_Original"], inplace=True)

# ==========================================
# Save Cleaned Dataset
# ==========================================

df.to_csv("Chocolate_Sales_Cleaned.csv", index=False)

print("\nCleaned dataset saved as 'Chocolate_Sales_Cleaned.csv'")

print("\n" + "=" * 60)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 60)