# ==========================================
# STEP 10 : HYPERPARAMETER TUNING
# ==========================================

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width",150)

print("="*60)
print("STEP 10 : HYPERPARAMETER TUNING")
print("="*60)

train_df = pd.read_csv("Train_Data.csv")
test_df = pd.read_csv("Test_Data.csv")

X_train = train_df.drop("Amount", axis=1)
y_train = train_df["Amount"]

X_test = test_df.drop("Amount", axis=1)
y_test = test_df["Amount"]

print("\nDataset Loaded Successfully")

param_grid = {

    "n_estimators":[100,200],

    "max_depth":[10,20,None],

    "min_samples_split":[2,5],

    "min_samples_leaf":[1,2],

    "max_features":["sqrt","log2"]

}

rf = RandomForestRegressor(
    random_state=42
)

grid_search = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    cv=3,

    scoring="r2",

    n_jobs=-1,

    verbose=2

)

grid_search.fit(X_train,y_train)

print("\nBest Parameters")

print(grid_search.best_params_)

best_rf = grid_search.best_estimator_

predictions = best_rf.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nTuned Model Performance")

print("-"*40)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

print("\nPrevious Random Forest")

print("R2 : 0.9021")

print("\nTuned Random Forest")

print(f"R2 : {r2:.4f}")

