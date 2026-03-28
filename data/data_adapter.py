import pandas as pd
import pathlib


def load_training_dataframe():
    """loads the training dataframe and cleans data for training"""

    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "logistics_shipments_dataset.csv"

    df = pd.read_csv(DATA_PATH)

    # refine data to only delivered shipments to avoid uncertain outcomes
    df = df[df["Status"] == "Delivered"]

    # strip whitespaces
    df['Origin_Warehouse'] = df['Origin_Warehouse'].str.strip()
    df['Destination'] = df['Destination'].str.strip()
    df['Carrier'] = df['Carrier'].str.strip()
    df['Service'] = df['Service'].str.strip()

    # ensure numeric datatypes
    df['Weight_kg'] = pd.to_numeric(df['Weight_kg'], errors='coerce')
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
    df['Distance_Miles'] = pd.to_numeric(df['Distance_Miles'], errors='coerce')

    # drop NA values from training columns
    df = df.dropna(
        subset=['Origin_Warehouse', 'Destination', 'Carrier', 'Weight_kg', 'Cost', 'Service', 'Distance_Miles'])

    # remove negative amounts
    df = df[df['Cost'] > 0]
    df = df[df['Weight_kg'] > 0]
    df = df[df['Distance_Miles'] > 0]


    # cap extreme cost values to reduce noise from simulated outliers
    # upper = df["Cost"].quantile(0.95)
    # df.loc[:, "Cost"] = df["Cost"].clip(upper=upper)

    return df

