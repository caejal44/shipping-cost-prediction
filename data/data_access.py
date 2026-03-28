import pandas as pd
import pathlib


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "logistics_shipments_dataset.csv"


def load_delivered_shipments():
    """load, clean, and filter data to delivered shipments only (drops incomplete shipments)"""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Origin_Warehouse", "Destination", "Carrier", "Service", "Distance_Miles"])
    df = df[df["Status"] == "Delivered"]
    return df

def get_origin_warehouses(df):
    """filters unique origin warehouses from data"""
    origin_warehouses = df["Origin_Warehouse"].unique().tolist()
    origin_warehouses.sort()
    return origin_warehouses


def get_destination_stores(df):
    """filters unique destination stores (using city name) from data"""
    destination_stores = df["Destination"].unique().tolist()
    destination_stores.sort()
    return destination_stores


def get_eligible_carriers(df, origin, destination, service):
    """uses route and service information to select carriers that historically service the route"""
    carriers = (
        df[
            (df["Origin_Warehouse"] == origin) &
            (df["Destination"] == destination) &
            (df["Service"] == service)
        ]["Carrier"]
        .unique()
        .tolist()
    )
    carriers.sort()
    return carriers

def get_average_transit_days(df, origin, destination, service):
    """uses route information to average historical transit days for comparison to user input transit"""
    transit_days = df[
        (df["Origin_Warehouse"] == origin) &
        (df["Destination"] == destination) &
        (df["Service"] == service)
    ].groupby("Carrier", as_index=False)["Transit_Days"].mean().rename(columns={"Transit_Days": "Average_Transit_Days"})
    return transit_days

def get_average_cost(df, origin, destination, service, weight):
    """uses route information and weight range to average historical cost for comparison to prediction"""
    average_cost_df = df[
        (df["Origin_Warehouse"] == origin) &
        (df["Destination"] == destination) &
        (df["Service"] == service)]
    if weight <= 100:
        average_cost = (
            average_cost_df[(average_cost_df["Weight_kg"] >= 0) & (average_cost_df["Weight_kg"] <= 100)
            ].groupby("Carrier", as_index=False)["Cost"].mean().rename(columns={"Cost": "Average_Cost"})
        )
        return average_cost
    elif (weight > 100) and (weight <= 250):
        average_cost = (average_cost_df[(average_cost_df["Weight_kg"] > 100) & (average_cost_df["Weight_kg"] <= 250)
            ].groupby("Carrier", as_index=False)["Cost"].mean().rename(columns={"Cost": "Average_Cost"})
        )
        return average_cost
    elif (weight > 250) and (weight <= 350):
        average_cost = (average_cost_df[(average_cost_df["Weight_kg"] > 250) & (average_cost_df["Weight_kg"] <= 350)
        ].groupby("Carrier", as_index=False)["Cost"].mean().rename(
            columns={"Cost": "Average_Cost"})
            )
        return average_cost
    elif (weight > 350) and (weight <= 500):
        average_cost = (average_cost_df[(average_cost_df["Weight_kg"] > 350) & (average_cost_df["Weight_kg"] <= 500)
                                        ].groupby("Carrier", as_index=False)["Cost"].mean().rename(
            columns={"Cost": "Average_Cost"})
                        )
        return average_cost
    else:
        average_cost = (average_cost_df[(average_cost_df["Weight_kg"] > 500)
            ].groupby("Carrier", as_index=False)["Cost"].mean().rename(columns={"Cost": "Average_Cost"})
        )
        return average_cost

def get_services(df, origin, destination):
    """uses route information to select services that have been offered for the route"""
    services = (df[
            (df["Origin_Warehouse"] == origin) &
            (df["Destination"] == destination)
        ]["Service"].unique().tolist()
                )
    services.sort()
    return services

def get_miles(df, origin, destination):
    """uses route information to select miles for given route"""
    miles = (df[
        (df["Origin_Warehouse"] == origin) &
        (df["Destination"] == destination)]
        ["Distance_Miles"].unique().tolist()
    )
    if len(miles) == 0:
        raise ValueError("No miles found")
    elif len(miles) > 1:
        raise ValueError("Multiple miles found")
    else:
        miles = float(miles[0])
        return miles

def get_historical_volume(df, origin, destination, service):
    """uses route information to return historical volume by each carrier"""
    historical_volume = df[
        (df["Origin_Warehouse"] == origin) &
        (df["Destination"] == destination) &
        (df["Service"] == service)
    ].groupby("Carrier", as_index=False).size().rename(columns={"size": "Volume"})
    return historical_volume
