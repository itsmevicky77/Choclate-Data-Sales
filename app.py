# ==========================================
# Chocolate Sales Prediction API
# FastAPI
# ==========================================

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

import pandas as pd
import joblib


# ==========================================
# Load Saved Files
# ==========================================

model = joblib.load("models/best_model.pkl")

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

encoders = joblib.load(
    "models/label_encoders.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="Chocolate Sales Prediction API",
    version="1.0"
)

# Instrument the API for Prometheus metrics
Instrumentator().instrument(app).expose(app)

print("\nRegistered Routes")
print("=" * 50)

for route in app.routes:
    print(route.path)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(directory="templates")

# ==========================================
# Home Page
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={
        "prediction": None
    }
)

# ==========================================
# Prediction
# ==========================================

@app.post("/predict", response_class=HTMLResponse)
async def predict(

    request: Request,

    Product: str = Form(...),

    Country: str = Form(...),

    Channel: str = Form(...),

    Boxes_Shipped: int = Form(...),

    Price_per_Box: float = Form(...),

    Marketing_Spend: float = Form(...),

    Discount_Pct: float = Form(...),

    Month: int = Form(...),

    Quarter: int = Form(...)

):

    input_df = pd.DataFrame({

        "Boxes_Shipped":[Boxes_Shipped],

        "Price_per_Box":[Price_per_Box],

        "Marketing_Spend":[Marketing_Spend],

        "Discount_Pct":[Discount_Pct],

        "Product":[Product],

        "Country":[Country],

        "Channel":[Channel],

        "Month":[Month],

        "Quarter":[Quarter]

    })

    # Encode

    input_df["Product"] = encoders["Product"].transform(
        input_df["Product"]
    )

    input_df["Country"] = encoders["Country"].transform(
        input_df["Country"]
    )

    input_df["Channel"] = encoders["Channel"].transform(
        input_df["Channel"]
    )

    # Scale

    numeric_columns = [

        "Boxes_Shipped",

        "Price_per_Box",

        "Marketing_Spend",

        "Discount_Pct"

    ]

    input_df[numeric_columns] = scaler.transform(
        input_df[numeric_columns]
    )

    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)

    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={
        "prediction": round(float(prediction[0]), 2)
    }
)