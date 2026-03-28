
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
    load_delivered_shipments, get_services, get_miles, get_average_cost, get_historical_volume)

@st.cache_data
def load_data():
    return load_delivered_shipments()

@st.cache_resource
def load_model():
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "model" / "shipping_cost_pipeline.pkl"

    # ensure model exists
    if not MODEL_PATH.exists():
        train_once()

    # load once and cache
    return joblib.load(MODEL_PATH)

#--------------------------------
# App Setup
#--------------------------------

st.title("Shipment Cost Predictor")
st.write("Estimate and compare shipping costs across carriers")

# ensures model has been trained and pipeline file loaded
pipeline = load_model()

# load historical data
shipment_df = load_data()

#--------------------------------
# Sidebar Setup
#--------------------------------

# sidebar is used to collect user input for model consumption
origin = st.sidebar.selectbox("Origin Warehouse", get_origin_warehouses(shipment_df))
destination = st.sidebar.selectbox("Destination Store", get_destination_stores(shipment_df))
weight = st.sidebar.number_input("Shipment Weight (kgs)", min_value = 1.0, max_value = 5000.0, value=25.0)
service = st.sidebar.selectbox("Service", get_services(shipment_df, origin, destination))
delivery = st.sidebar.date_input("Select desired delivery date", datetime.date.today())
predict = st.sidebar.button("Predict Now")

#--------------------------------
# Prediction Logic
#--------------------------------

# Once predict button has been submitted, desired transit is calculated and carriers who have
# historically serviced the route are discovered.
if predict:
    transit_days = max((delivery - datetime.date.today()).days,1)
    carriers = get_eligible_carriers(shipment_df, origin, destination, service)
    if not carriers:
        st.warning("No carriers available for this route and service.")
        st.info("Try selecting a different service level or route.")
        st.stop()
    miles = get_miles(shipment_df, origin, destination)
    results = []

    # user input is added to historically serviced carriers, fed to pipeline,
    # and estimates stored in results
    for carrier in carriers:
        shipment = {
            "Origin_Warehouse": origin,
            "Destination": destination,
            "Carrier": carrier,
            "Weight_kg": weight,
            "Service": service,
            "Distance_Miles": miles
        }
        cost = predict_cost(pipeline, shipment)
        results.append({"Carrier": carrier, "Estimated Cost": cost})

    # results are converted to DataFrame for sorting and display
    results_df = pd.DataFrame(results)

    # route information is used to calculate historical transit for comparison to desired transit
    average_transit_days_df = get_average_transit_days(shipment_df, origin, destination, service)

    # route information and weight is used to calculate average cost for comparison to prediction
    average_cost_df = get_average_cost(shipment_df, origin, destination, service, weight)

    # route information  is used to pull historical volume
    historical_volume_df = get_historical_volume(shipment_df, origin, destination, service)


    # joins average transit to results DataFrame, ranks results by cost,
    # and flags delivery risk based on historical transit
    final_results_df = pd.merge(results_df, average_transit_days_df, on=["Carrier"], how="left")
    final_results_df = final_results_df.sort_values(by = "Estimated Cost")
    final_results_df = final_results_df.reset_index(drop = True)
    final_results_df.insert(0, "Rank", final_results_df.index + 1)
    warning_icon = "⚠️"
    check_icon = "✅"
    final_results_df["Transit Status"] = np.where(final_results_df["Average_Transit_Days"] > transit_days,
        f"{warning_icon}At Risk of Late Delivery",
        f"{check_icon}On-Time Delivery")

#--------------------------------
# Display Logic
#--------------------------------

    # displays user input via DataFrame
    st.subheader("Shipment Summary (Your Information)")
    st.dataframe({"Origin": [origin],
    "Destination": [destination],
    "Weight": [weight],
    "Service": [service],
    "Delivery Date": [delivery],
    "Transit Days": [transit_days]},
    column_config={"Delivery Date": st.column_config.DateColumn(format="MM/DD/YYYY")}, hide_index=True)

    # adds display carrier column to show top ranked carrier
    top_choice = final_results_df.loc[final_results_df["Rank"] == 1, "Carrier"].iloc[0]
    final_results_df["Display Carrier"] = final_results_df["Carrier"].apply(
        lambda x: f"{x} (*)" if x == top_choice else x)
    final_results_df_columns = final_results_df[["Carrier", "Display Carrier", "Estimated Cost"]]

    # displays and ranks predictions based on estimated cost
    st.subheader("Carrier Rankings")
    st.write(f"Route: {origin} -> {destination} | Service: {service}")
    st.caption("Click any column to sort results")
    st.dataframe(final_results_df, column_config={"Estimated Cost":
        st.column_config.NumberColumn("Estimated Cost ($)", format="$%.2f"),
            "Average_Transit_Days": st.column_config.NumberColumn("Average Transit (Days)", format="%.1f"),
                    "Display Carrier": "Carrier"},
                 hide_index=True, column_order=("Rank", "Display Carrier", "Estimated Cost", "Average_Transit_Days", "Transit Status"))


    # --------------------------------
    # Visual Option Selections
    # --------------------------------

    st.caption("Compare carriers across speed, cost, and data coverage.")
    tab1, tab2, tab3 = st.tabs(["Transit Time", "Cost Benchmark", "Data Coverage"])

    # --------------------------------
    # Bar Chart - Transit
    # --------------------------------
    with tab1:
        # displays historical average transit for the selected service as a bar chart
        st.subheader("Average Transit by Carrier")
        st.caption("(*) - indicates #1 ranked carrier")

        fig, ax = plt.subplots()
        final_results_df_sorted = final_results_df.sort_values(by = "Average_Transit_Days", ascending=True)
        bars = ax.bar(final_results_df_sorted["Display Carrier"], final_results_df_sorted["Average_Transit_Days"])
        ax.set_ylabel("Transit Days")
        plt.axhline(y=transit_days, color="red", linestyle="--", label=f"Delivery Target: {transit_days} Days")
        ax.bar_label(bars, fmt="%.1f", padding=5)
        ax.legend(fontsize=8)
        plt.xticks(rotation=15, ha="right", fontsize=6)
        plt.yticks(fontsize=7)
        st.pyplot(fig)

    # --------------------------------
    # Bar Chart - Average Cost
    # --------------------------------
    with tab2:
        # displays historical average cost for the selected service as a bar chart
        st.subheader("Average Cost by Carrier")
        st.caption("Average cost is based on shipments within the same weight range:")
        st.caption("1-100 kgs | 101-250 kgs | 251-350 kgs | 351-500 kgs | 501+ kgs")
        st.caption("(*) - indicates #1 ranked carrier")

        if average_cost_df.empty:
            st.warning("No historical cost data available for this route, service, and weight range.")
        else:
            fig, ax = plt.subplots()
            average_cost_df = pd.merge(average_cost_df, final_results_df_columns, on=["Carrier"], how="left")
            average_cost_df_sorted = average_cost_df.sort_values(by = "Average_Cost", ascending=True)
            bars = ax.bar(average_cost_df_sorted["Display Carrier"], average_cost_df_sorted["Average_Cost"], label="Average Cost")
            ax.set_ylabel("Average Cost")
            ax.bar_label(bars, fmt="%.2f", padding=5)
            ax.plot(average_cost_df_sorted["Display Carrier"], average_cost_df_sorted["Estimated Cost"],
                         color="red", marker="o", linestyle="--", label="Estimated Cost")
            plt.xticks(rotation=15, ha="right", fontsize=6)
            plt.yticks(fontsize=7)
            ax.legend(fontsize=6)
            st.pyplot(fig)

    # --------------------------------
    # Bar Chart - Historical Volume by Carrier
    # --------------------------------

    with tab3:
        # displays historical average cost for the selected service as a bar chart
        st.subheader("Historical Shipment Volume (Data Coverage)")
        st.caption("Higher shipment volume indicates stronger historical data support and more reliable estimates.")
        st.caption(f"Route: {origin} -> {destination} | Service: {service}")
        st.caption("(*) - indicates #1 ranked carrier")

        if historical_volume_df.empty:
            st.warning("No historical volume data available for this route and service.")
        else:
            fig, ax = plt.subplots()
            historical_volume_df = pd.merge(historical_volume_df, final_results_df_columns, on=["Carrier"], how="left")
            historical_volume_df_sorted = historical_volume_df.sort_values(by="Volume", ascending=False)
            bars = ax.bar(historical_volume_df_sorted["Display Carrier"], historical_volume_df_sorted["Volume"])
            ax.set_ylabel("Historical Volume")
            ax.bar_label(bars)
            plt.xticks(rotation=15, ha="right", fontsize=6)
            plt.yticks(fontsize=7)
            st.pyplot(fig)



    # disclaimer
    st.write("Estimates are based on historical data and may vary due to carrier pricing and service conditions.")
