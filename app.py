import streamlit as st
import pandas as pd
import datetime

st.set_page_config(page_title="LOOP AI Demand Forecast Dashboard", layout="centered")
st.title("🚖 LOOP AI Demand Forecast Dashboard")

# File uploader section
uploaded_file = st.file_uploader("📁 Upload your AI forecast CSV", type=["csv"])

if uploaded_file:
    # Read uploaded CSV
    demand_df = pd.read_csv(uploaded_file, parse_dates=["time_block"])

    # Show preview
    st.subheader("🔍 Data Preview")
    st.dataframe(demand_df.head())

    # Identify tomorrow's date from data
    latest_date = demand_df['time_block'].max().date()
    tomorrow_date = latest_date + pd.Timedelta(days=1)

    # Predict high-demand zones (yhat > 3)
    high_demand_zones = []

    grouped = demand_df.groupby('geosquare')
    for geosquare, group in grouped:
        tomorrow_forecast = group[group['time_block'].dt.date == tomorrow_date]
        high_demand_times = tomorrow_forecast[tomorrow_forecast['yhat'] > 3]['time_block'].tolist()
