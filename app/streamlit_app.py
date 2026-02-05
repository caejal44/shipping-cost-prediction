
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import joblib

from model.predicting import predict_cost
from model.training import train_once
from data.data_access import (
    get_eligible_carriers,
    get_origin_warehouses,
    get_destination_stores,
    get_average_transit_days,
    load_delivered_shipments)

#--------------------------------
# App Setup
#--------------------------------

st.title("Shipment Cost Predictor")
st.caption("Estimates and ranks price per carrier based on anticipated shipment details")

# ensures model has been trained and pipeline file loaded
train_once()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "shipping_cost_pipeline.pkl"
pipeline = joblib.load(MODEL_PATH)

#--------------------------------
# Sidebar Setup
#--------------------------------

# sidebar is used to collect user input for model consumption
origin = st.sidebar.selectbox("Origin Warehouse", get_origin_warehouses())
destination = st.sidebar.selectbox("Destination Store", get_destination_stores())
weight = st.sidebar.number_input("Shipment Weight (kgs)", min_value = 1.0, max_value = 5000.0, value=25.0)
delivery = st.sidebar.date_input("Select desired delivery date", datetime.date.today())
predict = st.sidebar.button("Predict Now")

#--------------------------------
# Prediction Logic
#--------------------------------

# Once predict button has been submitted, desired transit is calculated and carriers who have
# historically serviced the route are discovered.
if predict:
    transit_days = max((delivery - datetime.date.today()).days,1)
    carriers = get_eligible_carriers(origin, destination)
    results = []

    # user input is added to historically serviced carriers, fed to pipeline,
    # and estimates stored in results
    for carrier in carriers:
        shipment = {
            "Origin_Warehouse": origin,
            "Destination": destination,
            "Carrier": carrier,
            "Weight_kg": weight,
            "Transit_Days": transit_days,
        }
        cost = predict_cost(pipeline, shipment)
        results.append({"Carrier": carrier, "Estimated Cost": cost})

    # results are converted to DataFrame for sorting and display
    results_df = pd.DataFrame(results)

    # route information is used to calculate historical transit for comparison to desired transit
    average_transit_days_df = get_average_transit_days(origin, destination)


    # joins average transit to results DataFrame, ranks results by cost,
    # and flags delivery risk based on historical transit
    final_results_df = pd.merge(results_df, average_transit_days_df, on=["Carrier"], how="left")
    final_results_df = final_results_df.sort_values(by = "Estimated Cost")
    final_results_df = final_results_df.reset_index(drop = True)
    final_results_df.insert(0, "Rank", final_results_df.index + 1)
    final_results_df["Status"] = np.where(final_results_df["Average_Transit_Days"] > transit_days,
        "Warning - Avg transit exceeds delivery goal",
        "Delivery goal within avg transit")

#--------------------------------
# Display Logic
#--------------------------------

    # displays user input via DataFrame
    st.header("Shipment Summary (Your Information)")
    st.dataframe({"Origin": [origin],
    "Destination": [destination],
    "Weight": [weight],
    "Transit Days": [transit_days]}, hide_index=True)

    # displays and ranks predictions based on estimated cost
    st.header("Carrier Rankings")
    st.dataframe(final_results_df, column_config={"Estimated Cost":
        st.column_config.NumberColumn("Estimated Cost ($)", format="$%.2f")}, hide_index=True)

    # --------------------------------
    # Bar Chart
    # --------------------------------

    # converts prediction into a horizontal bar chart with prices displayed
    st.header("Estimated Cost by Carrier")
    fig, ax = plt.subplots()
    bars = ax.barh(final_results_df["Carrier"], final_results_df["Estimated Cost"])
    ax.bar_label(bars, labels=[f'${x:,.2f}' for x in bars.datavalues], padding=3)
    ax.set_ylabel("Carrier")
    ax.set_xlabel("Estimated Cost")
    st.pyplot(fig)

#--------------------------------
# Box Plot
#--------------------------------

    # shows historical cost ranges as a boxplot for each carrier that previously serviced this route
    st.header("Historical Cost Distribution by Carrier (Selected Route)")
    df = load_delivered_shipments()
    route_df = df[
    (df["Origin_Warehouse"] == origin) &
    (df["Destination"] == destination)
    ]
    fig, ax = plt.subplots()
    route_df.boxplot(column="Cost", by="Carrier", ax=ax)
    st.pyplot(fig)

#--------------------------------
# Scatter Plot
#--------------------------------

    # shows historical weight vs. cost at 95th percentile (both) for the route
    # as a scatter plot with carriers color-coded
    st.header("Historical Shipment Cost vs. Weight from Origin Location")
    st.caption("Scatter plot excludes the top 5% to improve readability")
    scatter_df = df[(df["Origin_Warehouse"]) == origin]

    plot_df = scatter_df.copy()
    cost_cap = plot_df["Cost"].quantile(0.95)
    weight_cap = plot_df["Weight_kg"].quantile(0.95)
    plot_df = plot_df[(plot_df["Cost"] < cost_cap) & (plot_df["Weight_kg"] < weight_cap)]

    fig, ax = plt.subplots()
    for carrier, group in plot_df.groupby("Carrier"):
        ax.scatter(
            group["Weight_kg"],
            group["Cost"],
            alpha=0.3,
            label=carrier
        )

    ax.legend(title="Carrier", fontsize=8)
    ax.set_xlabel("Shipment Weight (kg)")
    ax.set_ylabel("Historical Shipment Cost")
    st.pyplot(fig)

    # disclaimer
    st.write("Estimates are based on historical shipment data and may vary based on service "
             "conditions and carrier pricing.")
