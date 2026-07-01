# 🍫 Chocolate Sales Prediction Dashboard

> An end-to-end Machine Learning project that predicts chocolate sales revenue using historical sales data and deploys the model through a modern FastAPI web dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

---

# 📌 Project Overview

This project demonstrates a complete **Machine Learning workflow** for predicting chocolate sales revenue using historical business data.

The project includes:

- Data Understanding
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Selection
- Data Preprocessing
- Model Building
- Hyperparameter Tuning
- Model Evaluation
- Model Selection
- Model Saving
- Inference Pipeline
- FastAPI Deployment
- Interactive Dashboard

---

# 🚀 Dashboard Preview

> Add screenshots here after uploading them.

### Dashboard

```
images/dashboard.png
```

### Prediction Result

```
images/prediction.png
```

---

# 🎯 Problem Statement

Businesses need accurate revenue forecasting to improve:

- Inventory Planning
- Marketing Budget Allocation
- Supply Chain Decisions
- Sales Strategy
- Demand Forecasting

This project predicts the expected sales revenue based on product, location, pricing, discounts, marketing spend, and shipment details.

---

# 📂 Dataset

Dataset contains approximately:

| Description | Value |
|------------|-------|
| Total Records | 194,754 |
| Features | 9 Selected Features |
| Target Variable | Amount |
| Train-Test Split | 80 : 20 |

---

# 🧹 Data Preprocessing Pipeline

The preprocessing pipeline includes:

- Missing Value Handling
- Date Conversion
- Amount Cleaning
- Feature Encoding
- Feature Scaling
- Train-Test Split

---

# 📊 Exploratory Data Analysis

EDA includes:

- Product Distribution
- Country Distribution
- Channel Distribution
- Salesperson Distribution
- Revenue Analysis
- Summary Statistics

---

# ⚙️ Feature Engineering

Created new features including:

- Year
- Month
- Quarter
- Day
- Day of Week
- Week of Year
- Weekend Indicator
- Discount Category
- Price Category
- Marketing Efficiency
- Revenue Per Box

---

# 🔍 Feature Selection

Feature selection was performed using:

- Correlation Analysis
- Mutual Information
- Random Forest Feature Importance

Final Selected Features:

- Boxes Shipped
- Price per Box
- Marketing Spend
- Discount Percentage
- Product
- Country
- Channel
- Month
- Quarter

---

# 🤖 Machine Learning Models

The following regression models were trained and compared:

| Model | R² Score | MAE | RMSE |
|--------|---------:|----:|-----:|
| Linear Regression | 0.7063 | 100.40 | 200.35 |
| Decision Tree | 0.8058 | 45.85 | 162.92 |
| Random Forest | 0.9021 | 33.33 | 115.66 |
| Tuned Random Forest | ⭐ 0.9051 | 37.52 | 113.87 |

### ✅ Best Model

Random Forest Regressor

---

# 📈 Model Performance

| Metric | Score |
|---------|-------|
| R² Score | **0.9051** |
| MAE | **37.52** |
| RMSE | **113.87** |

---

# 🛠 Tech Stack

### Programming

- Python

### Libraries

- Pandas
- NumPy
- Scikit-Learn
- Joblib

### Visualization

- Matplotlib
- Seaborn

### Deployment

- FastAPI
- HTML
- CSS
- JavaScript
- Jinja2

---

# 📁 Project Structure

```
Chocolate_Data_Sales/
│
├── app.py
├── Inference.py
├── Save_Model.py
├── Model_Building.py
├── Hyperparameter_Tuning.py
├── Model_Evaluation.py
├── Model_Selection.py
├── Feature_Engineering.py
├── Feature_Selection.py
├── Data_Preprocessing.py
├── Data_cleaning.py
├── Dat_Understanding.py
├── EDA.py
│
├── models/
│   ├── feature_columns.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
│
├── templates/
│
├── static/
│
├── images/
│
├── Train_Data.csv
├── Test_Data.csv
│
├── Chocolate_Sales.csv
├── Chocolate_Sales_Cleaned.csv
├── Chocolate_Sales_Feature_Engineered.csv
├── Chocolate_Sales_Selected_Features.csv
│
└── README.md
```

---

# 💻 Installation

Clone the repository

```bash
git clone git@github.com:itsmevicky77/Choclate_Data_Sales.git
```

Move into the project folder

```bash
cd Choclate_Data_Sales
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Run FastAPI

```bash
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

# 🔮 Future Improvements

- Docker Deployment
- Streamlit Dashboard
- MLflow Integration
- DVC Pipeline
- Cloud Deployment
- CI/CD using GitHub Actions
- Power BI Dashboard

---

# 👨‍💻 Author

## Vignesh Narayanan

Data Science & AI Engineer

### Skills

- Python
- Machine Learning
- FastAPI
- Data Analytics
- Scikit-Learn
- SQL
- Pandas

---

# 📜 License

This project is licensed under the MIT License.

---

## ⭐ If you found this project useful, consider giving it a Star!