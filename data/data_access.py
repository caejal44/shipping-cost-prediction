import pandas as pd
import pathlib


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "logistics_shipments_dataset.csv"


def load_delivered_shipments():
    """load, clean, and filter data to delivered shipments only (drops incomplete shipments)"""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Origin_Warehouse", "Destination", "Carrier"])
    df = df[df["Status"] == "Delivered"]
    return df

def get_origin_warehouses():
    """filters unique origin warehouses from data"""
    df = load_delivered_shipments()
    origin_warehouses = df["Origin_Warehouse"].unique().tolist()
    origin_warehouses.sort()
    return origin_warehouses


def get_destination_stores():
    """filters unique destination stores (using city name) from data"""
    df = load_delivered_shipments()
    destination_stores = df["Destination"].unique().tolist()
    destination_stores.sort()
    return destination_stores


def get_eligible_carriers(origin, destination):
    """uses route information to select carriers that historically service the route"""
    df = load_delivered_shipments()
    carriers = (
        df[
            (df["Origin_Warehouse"] == origin) &
            (df["Destination"] == destination)
        ]["Carrier"]
        .unique()
        .tolist()
    )
    return carriers

def get_average_transit_days(origin, destination):
    """uses route information to average historical transit days for comparison to user input transit"""
    df = load_delivered_shipments()
    transit_days = df[
        (df["Origin_Warehouse"] == origin) &
        (df["Destination"] == destination)
    ].groupby("Carrier", as_index=False)["Transit_Days"].mean().rename(columns={"Transit_Days": "Average_Transit_Days"})
    return transit_days