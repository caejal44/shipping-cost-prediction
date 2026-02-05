import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pathlib
from pathlib import Path


def train_model():
    """Trains a random forest model to estimate shipping cost using historical shipment data. Extreme cost values
    are capped at the 95th percentile to reduce noise from simulated outliers.

    Categorical features:
    Origin_Warehouse
    Destination
    Carrier

    Numerical features:
    Weight
    Transit_Days (proxy for service level and delivery speed)

    Target variable:
    Cost

    Returns:
    sklearn.pipeline.Pipeline: Trained regression pipeline
    """

    # import historical data and remove missing values
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "logistics_shipments_dataset.csv"

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['Origin_Warehouse', 'Destination', 'Carrier', 'Weight_kg', 'Transit_Days', 'Cost'])

    # further refine data to only delivered shipments to avoid uncertain outcomes and invalid transit days
    df = df[df["Status"] == "Delivered"]

    # cap extreme cost values to reduce noise from simulated outliers
    upper = df["Cost"].quantile(0.95)
    df.loc[:, "Cost"] = df["Cost"].clip(upper=upper)

    # split target and training data
    y = df['Cost']
    X = df[['Origin_Warehouse', 'Destination', 'Carrier', 'Weight_kg', 'Transit_Days']]
    categorical = ['Origin_Warehouse', 'Destination', 'Carrier']
    numerical = ['Weight_kg', 'Transit_Days']

    # set encoding for categorical data
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical), ('num', 'passthrough', numerical)])

    # build random forest model and pipeline
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ]
    )

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipeline.fit(X_train, y_train)

    # evaluate model
    y_pred = pipeline.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))

    return pipeline

def train_once():

    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "model" / "shipping_cost_pipeline.pkl"

    if MODEL_PATH.exists():
        print("Model already trained. No training necessary.")
        return MODEL_PATH
    else:
        pipeline = train_model()
        joblib.dump(pipeline, MODEL_PATH)
        print("Model now trained successfully")
        return pipeline











