# ==========================================
# STEP 4 : EXPLORATORY DATA ANALYSIS (EDA)
# Part 1 - Dataset Overview
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("=" * 60)
print("STEP 4 : EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ==========================================
# Load Cleaned Dataset
# ==========================================

df = pd.read_csv("Chocolate_Sales_Cleaned.csv")

# Convert Order_Date back to datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("\nDataset Loaded Successfully")
print(f"Shape : {df.shape}")

# ==========================================
# Dataset Information
# ==========================================

print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)

print(df.info())

# ==========================================
# Data Types
# ==========================================

print("\nData Types")
print(df.dtypes)

# ==========================================
# Missing Values
# ==========================================

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================
# Duplicate Records
# ==========================================

print("\nDuplicate Records")
print(df.duplicated().sum())

# ==========================================
# Statistical Summary
# ==========================================

print("\nNumerical Summary")
print(df.describe())

print("\nCategorical Summary")
print(df.describe(include=["object"]))

# ==========================================
# Total Revenue
# ==========================================

print("\n" + "="*60)
print("BUSINESS OVERVIEW")
print("="*60)

print(f"Total Revenue       : ${df['Amount'].sum():,.2f}")
print(f"Total Orders        : {len(df):,}")
print(f"Total Boxes Sold    : {df['Boxes_Shipped'].sum():,}")
print(f"Average Order Value : ${df['Amount'].mean():,.2f}")

print("\nEDA PART 1 COMPLETED")

# ==========================================
# EDA PART 2 - UNIVARIATE ANALYSIS
# ==========================================

import os

# Create images folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# ==========================================
# Product Distribution
# ==========================================

print("\nCreating Product Distribution Chart...")

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    y="Product",
    order=df["Product"].value_counts().index
)

plt.title("Product Distribution")
plt.xlabel("Number of Orders")
plt.ylabel("Product")

plt.tight_layout()

plt.savefig("images/product_distribution.png")

plt.show()

# ==========================================
# Country Distribution
# ==========================================

print("\nCreating Country Distribution Chart...")

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Country",
    order=df["Country"].value_counts().index
)

plt.title("Country Distribution")

plt.tight_layout()

plt.savefig("images/country_distribution.png")

plt.show()

# ==========================================
# Sales Channel Distribution
# ==========================================
print("\nCreating Channel Distribution Chart...")

plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="Channel",
    order=df["Channel"].value_counts().index
)

plt.title("Sales Channel Distribution")

plt.tight_layout()

plt.savefig("images/channel_distribution.png")

plt.show()

# ==========================================
# Salesperson Distribution
# ==========================================

print("\nCreating Salesperson Distribution Chart...")

plt.figure(figsize=(14,8))

sns.countplot(
    data=df,
    y="Salesperson",
    order=df["Salesperson"].value_counts().index
)

plt.title("Salesperson Distribution")

plt.tight_layout()

plt.savefig("images/salesperson_distribution.png")

plt.show()

# ==========================================
# Revenue by Product
# ==========================================

print("\nCreating Revenue by Product...")

product_sales = (
    df.groupby("Product")["Amount"]
      .sum()
      .sort_values(ascending=False)
)

plt.figure(figsize=(12,6))

sns.barplot(
    x=product_sales.values,
    y=product_sales.index
)

plt.title("Revenue by Product")
plt.xlabel("Revenue ($)")
plt.ylabel("Product")

plt.tight_layout()

plt.savefig("images/revenue_by_product.png")

plt.show()

print(product_sales)

# ==========================================
# Revenue by Country
# ==========================================

country_sales = (
    df.groupby("Country")["Amount"]
      .sum()
      .sort_values(ascending=False)
)

plt.figure(figsize=(8,5))

sns.barplot(
    x=country_sales.index,
    y=country_sales.values
)

plt.title("Revenue by Country")
plt.ylabel("Revenue ($)")

plt.tight_layout()

plt.savefig("images/revenue_by_country.png")

plt.show()

print(country_sales)

# ==========================================
# Revenue by Sales Channel
# ==========================================

channel_sales = (
    df.groupby("Channel")["Amount"]
      .sum()
      .sort_values(ascending=False)
)

plt.figure(figsize=(6,5))

sns.barplot(
    x=channel_sales.index,
    y=channel_sales.values
)

plt.title("Revenue by Sales Channel")

plt.tight_layout()

plt.savefig("images/revenue_by_channel.png")

plt.show()

print(channel_sales)

# ==========================================
# Top 10 Salespersons by Revenue
# ==========================================

salesperson_sales = (
    df.groupby("Salesperson")["Amount"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=salesperson_sales.values,
    y=salesperson_sales.index
)

plt.title("Top 10 Salespersons by Revenue")

plt.tight_layout()

plt.savefig("images/top_salespersons_revenue.png")

plt.show()

print(salesperson_sales)

# ==========================================
# Print Summary Statistics
# ==========================================

print("\nTop Products")
print(df["Product"].value_counts())

print("\nCountry Distribution")
print(df["Country"].value_counts())

print("\nSales Channel Distribution")
print(df["Channel"].value_counts())

print("\nTop Salespersons")
print(df["Salesperson"].value_counts())

# ==========================================
# EDA PART 4 - TIME SERIES ANALYSIS
# ==========================================

print("\n" + "=" * 60)
print("TIME SERIES ANALYSIS")
print("=" * 60)

# Create Date Features
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.strftime("%b")
df["Quarter"] = df["Order_Date"].dt.quarter

print("\nTime Features Created Successfully")

monthly_sales = (
    df.groupby(["Year", "Month", "Month_Name"])["Amount"]
      .sum()
      .reset_index()
      .sort_values(["Year", "Month"])
)

plt.figure(figsize=(14,6))

plt.plot(
    range(len(monthly_sales)),
    monthly_sales["Amount"],
    marker="o",
    linewidth=2
)

plt.xticks(
    range(len(monthly_sales)),
    monthly_sales["Month_Name"] + "-" + monthly_sales["Year"].astype(str),
    rotation=45
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue ($)")

plt.tight_layout()

plt.savefig("images/monthly_revenue_trend.png")

plt.show()

print(monthly_sales)

quarterly_sales = (
    df.groupby(["Year", "Quarter"])["Amount"]
      .sum()
      .reset_index()
)

plt.figure(figsize=(8,5))

sns.barplot(
    data=quarterly_sales,
    x="Quarter",
    y="Amount",
    hue="Year"
)

plt.title("Quarterly Revenue")

plt.tight_layout()

plt.savefig("images/quarterly_revenue.png")

plt.show()

print(quarterly_sales)

yearly_sales = (
    df.groupby("Year")["Amount"]
      .sum()
)

plt.figure(figsize=(6,5))

sns.barplot(
    x=yearly_sales.index.astype(str),
    y=yearly_sales.values
)

plt.title("Yearly Revenue")

plt.tight_layout()

plt.savefig("images/yearly_revenue.png")

plt.show()

print(yearly_sales)

monthly_boxes = (
    df.groupby(["Year", "Month", "Month_Name"])["Boxes_Shipped"]
      .sum()
      .reset_index()
      .sort_values(["Year", "Month"])
)

plt.figure(figsize=(14,6))

plt.plot(
    range(len(monthly_boxes)),
    monthly_boxes["Boxes_Shipped"],
    marker="o"
)

plt.xticks(
    range(len(monthly_boxes)),
    monthly_boxes["Month_Name"] + "-" + monthly_boxes["Year"].astype(str),
    rotation=45
)

plt.title("Monthly Boxes Shipped")

plt.tight_layout()

plt.savefig("images/monthly_boxes_shipped.png")

plt.show()

monthly_sales["Growth_%"] = monthly_sales["Amount"].pct_change() * 100

plt.figure(figsize=(14,6))

plt.plot(
    range(len(monthly_sales)),
    monthly_sales["Growth_%"],
    marker="o"
)

plt.axhline(0, linestyle="--")

plt.xticks(
    range(len(monthly_sales)),
    monthly_sales["Month_Name"] + "-" + monthly_sales["Year"].astype(str),
    rotation=45
)

plt.title("Monthly Revenue Growth (%)")

plt.tight_layout()

plt.savefig("images/monthly_growth.png")

plt.show()

print(monthly_sales[["Year","Month_Name","Growth_%"]])

# ==========================================
# EDA PART 5 - CORRELATION ANALYSIS
# ==========================================

print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

correlation = df[
    [
        "Discount_Pct",
        "Price_per_Box",
        "Marketing_Spend",
        "Boxes_Shipped",
        "Amount"
    ]
].corr()

print(correlation)

plt.figure(figsize=(8,6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("images/correlation_heatmap.png")

plt.show()

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Marketing_Spend",
    y="Amount",
    alpha=0.5
)

plt.title("Marketing Spend vs Revenue")

plt.tight_layout()

plt.savefig("images/marketing_vs_revenue.png")

plt.show()

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Discount_Pct",
    y="Amount",
    alpha=0.5
)

plt.title("Discount vs Revenue")

plt.tight_layout()

plt.savefig("images/discount_vs_revenue.png")

plt.show()

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Boxes_Shipped",
    y="Amount",
    alpha=0.5
)

plt.title("Boxes Shipped vs Revenue")

plt.tight_layout()

plt.savefig("images/boxes_vs_revenue.png")

plt.show()

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Price_per_Box",
    y="Amount",
    alpha=0.5
)

plt.title("Price per Box vs Revenue")

plt.tight_layout()

plt.savefig("images/price_vs_revenue.png")

plt.show()

sample_df = df.sample(3000, random_state=42)

sns.pairplot(
    sample_df[
        [
            "Discount_Pct",
            "Price_per_Box",
            "Marketing_Spend",
            "Boxes_Shipped",
            "Amount"
        ]
    ]
)

plt.savefig("images/pairplot.png")

plt.show()

print("\nCorrelation with Revenue")

print(
    correlation["Amount"]
    .sort_values(ascending=False)
)

# ==========================================
# EDA PART 6 - OUTLIER ANALYSIS
# ==========================================

print("\n" + "="*60)
print("OUTLIER ANALYSIS")
print("="*60)

numerical_columns = [
    "Discount_Pct",
    "Price_per_Box",
    "Marketing_Spend",
    "Boxes_Shipped",
    "Amount"
]

for column in numerical_columns:

    plt.figure(figsize=(8,5))

    sns.histplot(
        df[column],
        bins=40,
        kde=True
    )

    plt.title(f"Distribution of {column}")

    plt.tight_layout()

    plt.savefig(f"images/{column}_distribution.png")

    plt.show()

    print("\n" + "="*60)
print("OUTLIER DETECTION USING IQR")
print("="*60)

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    percentage = (len(outliers) / len(df)) * 100

    print(f"\n{column}")
    print(f"Lower Limit : {lower:.2f}")
    print(f"Upper Limit : {upper:.2f}")
    print(f"Outliers    : {len(outliers)}")
    print(f"Percentage  : {percentage:.2f}%")

summary = []

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    summary.append({
        "Feature": column,
        "Outliers": len(outliers),
        "Percentage": round((len(outliers)/len(df))*100,2)
    })

summary_df = pd.DataFrame(summary)

print("\nOutlier Summary")
print(summary_df)

# ==========================================
# EDA PART 7 - BUSINESS INSIGHTS
# ==========================================

print("\n" + "="*60)
print("BUSINESS INSIGHTS")
print("="*60)

top_product = (
    df.groupby("Product")["Amount"]
      .sum()
      .sort_values(ascending=False)
)

print("\nTop Revenue Product")
print(top_product.head(1))

top_country = (
    df.groupby("Country")["Amount"]
      .sum()
      .sort_values(ascending=False)
)

print("\nHighest Revenue Country")
print(top_country.head(1))

top_channel = (
    df.groupby("Channel")["Amount"]
      .sum()
      .sort_values(ascending=False)
)

print("\nBest Sales Channel")
print(top_channel)

top_salesperson = (
    df.groupby("Salesperson")["Amount"]
      .sum()
      .sort_values(ascending=False)
)

print("\nTop Salesperson")
print(top_salesperson.head(10))

monthly_best = (
    df.groupby(["Year","Month_Name"])["Amount"]
      .sum()
      .sort_values(ascending=False)
)

print("\nTop Revenue Months")
print(monthly_best.head(10))

print("\nAverage Discount")

print(
    round(df["Discount_Pct"].mean(),2),
    "%"
)

print("\nAverage Marketing Spend")

print(
    round(df["Marketing_Spend"].mean(),2)
)

print("\nAverage Order Value")

print(
    round(df["Amount"].mean(),2)
)

print("\n" + "="*60)
print("EXECUTIVE SUMMARY")
print("="*60)

print(f"""
1. Total Revenue Generated : ${df['Amount'].sum():,.2f}

2. Total Orders Processed : {len(df):,}

3. Total Boxes Shipped : {df['Boxes_Shipped'].sum():,}

4. Average Order Value : ${df['Amount'].mean():.2f}

5. Highest Revenue Product :
   {top_product.idxmax()}

6. Highest Revenue Country :
   {top_country.idxmax()}

7. Best Sales Channel :
   {top_channel.idxmax()}

8. Top Performing Salesperson :
   {top_salesperson.idxmax()}

9. Average Discount :
   {df['Discount_Pct'].mean():.2f} %

10. Average Marketing Spend :
    ${df['Marketing_Spend'].mean():.2f}

""")

print("="*60)
print("BUSINESS RECOMMENDATIONS")
print("="*60)

print("""
• Increase inventory for the highest revenue products.

• Allocate more marketing budget to the highest-performing countries.

• Analyze why the best sales channel performs well and replicate its strategies.

• Reward top-performing salespeople and use their sales techniques as best practices.

• Review products with low revenue and consider promotional campaigns or product improvements.

• Evaluate discount strategies to ensure they increase sales without significantly reducing profit.

• Continue investing in marketing activities that show a positive relationship with revenue.

• Monitor monthly and quarterly sales trends to improve inventory planning and demand forecasting.
""")