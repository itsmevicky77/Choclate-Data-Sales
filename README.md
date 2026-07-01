# 🍫 Chocolate Sales Dashboard
### AI-Powered Revenue Forecasting using Machine Learning, FastAPI & MLOps

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Docker](https://img.shields.io/badge/Docker-Container-blue?style=for-the-badge&logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge&logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800?style=for-the-badge&logo=grafana)

---

# 📌 Project Overview

The **Chocolate Sales Dashboard** is an end-to-end Machine Learning and MLOps project that predicts chocolate sales revenue based on product details, shipping information, pricing, discounts, marketing spend, and sales channels.

The project demonstrates the complete lifecycle of an ML application—from data preprocessing and model training to deployment, monitoring, and containerization.

---

# 🚀 Live Features

- 📈 Sales Revenue Prediction
- 🤖 Random Forest Regression Model
- ⚡ FastAPI REST API
- 🎨 Professional Interactive Dashboard
- 🐳 Docker Containerization
- 📊 MLflow Experiment Tracking
- 📡 Prometheus Monitoring
- 📉 Grafana Dashboard
- 🔄 GitHub Actions CI/CD Pipeline

---

# 📊 Dataset Information

| Attribute | Value |
|----------|--------|
| Domain | Chocolate Sales |
| Records | 194,754 |
| Target Variable | Amount |
| Selected Features | 9 |
| Dataset Type | Structured CSV |

### Input Features

- Product
- Country
- Sales Channel
- Month
- Quarter
- Discount %
- Boxes Shipped
- Price Per Box
- Marketing Spend

Target:

- Revenue (Amount)

---

# 🧠 Machine Learning Pipeline

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Feature Selection
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Selection
      │
      ▼
Save Best Model
      │
      ▼
FastAPI Deployment
      │
      ▼
Monitoring (Prometheus + Grafana)
```

---

# 🤖 Models Evaluated

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## ✅ Selected Model

Random Forest Regressor

---

# 📈 Model Performance

| Metric | Score |
|--------|--------|
| R² Score | **0.9051** |
| MAE | **33.33** |
| RMSE | **113.87** |

---

# 🖥 Dashboard

The dashboard allows users to:

- Select Product
- Select Country
- Choose Sales Channel
- Enter Month
- Enter Quarter
- Discount
- Boxes Shipped
- Price Per Box
- Marketing Spend

and instantly receive the predicted revenue.

---

# ⚙ Tech Stack

### Programming

- Python

### Machine Learning

- Pandas
- NumPy
- Scikit-Learn
- Joblib

### Backend

- FastAPI
- Uvicorn

### Frontend

- HTML
- CSS
- Jinja2

### MLOps

- Docker
- MLflow
- Prometheus
- Grafana
- GitHub Actions

---

# 📂 Project Structure

```
Chocolate_Sales/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── prometheus.yml
├── README.md
│
├── templates/
│      └── index.html
│
├── static/
│      └── style.css
│
├── models/
│
├── images/
│
├── evaluation_images/
│
├── .github/
│      └── workflows/
│
├── Data_Understanding.py
├── Data_Cleaning.py
├── EDA.py
├── Feature_Engineering.py
├── Feature_Selection.py
├── Data_Preprocessing.py
├── Model_Building.py
├── Hyperparameter_Tuning.py
├── Model_Evaluation.py
├── Model_Selection.py
├── Save_Model.py
├── Inference.py
└── MLflow_Tracking.py
```

---

# 🐳 Docker

Build the image

```bash
docker build -t chocolate-sales .
```

Run the container

```bash
docker run -p 8000:8000 chocolate-sales
```

Open

```
http://localhost:8000
```

---

# 📊 Monitoring

Prometheus collects:

- HTTP Requests
- Request Latency
- Response Time
- API Metrics

Grafana visualizes:

- Total Requests
- Requests Per Second
- Average Response Time
- Python Garbage Collection

---

# 📡 API Documentation

Swagger UI

```
http://localhost:8000/docs
```

Metrics

```
http://localhost:8000/metrics
```

---

# 🔄 CI/CD

GitHub Actions automatically:

- Installs dependencies
- Verifies project integrity
- Builds the application
- Ensures successful deployment workflow

---

# 🎯 Future Improvements

- Cloud Deployment (Render / Azure / AWS)
- Authentication
- Batch Predictions
- PostgreSQL Integration
- Real-time Data Pipeline
- Model Retraining Pipeline
- Kubernetes Deployment

---

# 👨‍💻 Developer

**Vignesh Narayanan**

AI & Machine Learning Enthusiast

- Python
- Machine Learning
- FastAPI
- MLOps
- Docker
- MLflow
- Prometheus
- Grafana

GitHub

https://github.com/itsmevicky77

LinkedIn

https://www.linkedin.com/in/vignesh-narayanan-13042b126/

---

# ⭐ If you found this project useful, consider giving it a Star!
