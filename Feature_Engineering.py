# ==========================================
# STEP 5 : FEATURE ENGINEERING
# ==========================================

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("="*60)
print("STEP 5 : FEATURE ENGINEERING")
print("="*60)

# Load Cleaned Dataset
df = pd.read_csv("Chocolate_Sales_Cleaned.csv")

# Convert Date
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("\nDataset Loaded Successfully")
print(df.shape)

print("\nCreating Date Features...")

df["Year"] = df["Order_Date"].dt.year

df["Month"] = df["Order_Date"].dt.month

df["Month_Name"] = df["Order_Date"].dt.month_name()

df["Quarter"] = df["Order_Date"].dt.quarter

df["Day"] = df["Order_Date"].dt.day

df["Day_of_Week"] = df["Order_Date"].dt.day_name()

df["Week_of_Year"] = df["Order_Date"].dt.isocalendar().week.astype(int)

df["Is_Weekend"] = df["Order_Date"].dt.dayofweek >= 5

print("Date Features Created Successfully")

print("\nCreating Revenue Per Box...")

df["Revenue_Per_Box"] = (
    df["Amount"] /
    df["Boxes_Shipped"]
)

print("Done")

print("\nCreating Marketing Efficiency...")

df["Marketing_Efficiency"] = (
    df["Amount"] /
    df["Marketing_Spend"]
)

print("Done")

print("\nCreating Discount Category...")

df["Discount_Category"] = pd.cut(

    df["Discount_Pct"],

    bins=[0,10,20,40],

    labels=["Low","Medium","High"],

    include_lowest=True

)

print(df["Discount_Category"].value_counts())

print("\nCreating Price Category...")

df["Price_Category"] = pd.qcut(

    df["Price_per_Box"],

    q=3,

    labels=["Low","Medium","High"]

)

print(df["Price_Category"].value_counts())

print("\nNew Dataset Shape")

print(df.shape)

print("\nNew Columns")

print(df.columns.tolist())

df.to_csv(
    "Chocolate_Sales_Feature_Engineered.csv",
    index=False
)

print("\nFeature Engineered Dataset Saved Successfully!")