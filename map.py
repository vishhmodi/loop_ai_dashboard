import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

st.set_page_config(page_title="LOOP AI Forecast Dashboard", layout="wide")
st.title("📊 LOOP AI Demand Forecast Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with latitude, longitude, and time", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check if required columns exist
    if set(['latitude', 'longitude', 'time']).issubset(df.columns):
        st.success("File uploaded successfully!")

        # Show data table
        st.subheader("📋 Uploaded Data")
        st.dataframe(df)

        # Create map
        st.subheader("🗺️ Heatmap of Predicted Demand Zones")
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
        heat_data = df[['latitude', 'longitude']].values.tolist()
        HeatMap(heat_data).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.error("CSV must contain 'latitude', 'longitude', and 'time' columns.")
else:
    st.info("Upload a CSV file to see predictions on the map.")
