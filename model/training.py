import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

from data.data_adapter import load_training_dataframe


def train_model():
    """Trains a random forest model to estimate shipping cost using historical shipment data.

    Categorical features:
    Origin_Warehouse
    Destination
    Carrier
    Service

    Numerical features:
    Weight
    Distance_Miles

    Target variable:
    Cost

    Returns:
    sklearn.pipeline.Pipeline: Trained regression pipeline
    """

    # load training data
    df = load_training_dataframe()

    # split target and training data
    y = df['Cost']
    X = df[['Origin_Warehouse', 'Destination', 'Carrier', 'Weight_kg', 'Service', 'Distance_Miles']]
    categorical = ['Origin_Warehouse', 'Destination', 'Carrier', 'Service']
    numerical = ['Weight_kg', 'Distance_Miles']

    # set encoding for categorical data
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical),
                      ('num', 'passthrough', numerical)])

    # build random forest model and pipeline
    rf_model = RandomForestRegressor(n_estimators=100,
                                     max_depth=15,
                                     min_samples_leaf=5,
                                     min_samples_split=10,
                                     random_state=42,
                                     n_jobs=-1)
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
    else:
        pipeline = train_model()
        joblib.dump(pipeline, MODEL_PATH, compress=3)
        print("Model now trained successfully")

    return MODEL_PATH

if __name__ == "__main__":
    train_once()










