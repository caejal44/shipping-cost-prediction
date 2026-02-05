import pandas as pd
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "logistics_shipments_dataset.csv"

def get_eligible_carriers(origin, destination):
    """uses the origin and destination to select carriers that historically serviced the route"""
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["Origin_Warehouse", "Destination", "Carrier"])
    df = df[df["Status"] == "Delivered"]

    carriers = (
        df[
            (df["Origin_Warehouse"] == origin) &
            (df["Destination"] == destination)
        ]["Carrier"]
        .unique()
        .tolist()
    )

    return carriers