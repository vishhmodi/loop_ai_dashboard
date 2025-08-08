import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="LOOP AI MVP", layout="wide")

st.title("LOOP AI Dashboard")
st.markdown("Upload your ride demand CSV file to analyze peak times and locations.")

# === File Upload ===
uploaded_file = st.file_uploader("Upload your demand CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Convert time to datetime and fix the format
    df['time'] = pd.to_datetime(df['time'])

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    # === Time Filtering ===
    min_time = df['time'].min().to_pydatetime()
    max_time = df['time'].max().to_pydatetime()
    time_range = st.slider(
        "🕒 Filter by Time Range",
        min_value=min_time,
        max_value=max_time,
        value=(min_time, max_time),
        format="MM/DD/YYYY - HH:mm"
    )

    filtered_df = df[(df['time'] >= time_range[0]) & (df['time'] <= time_range[1])]

    st.subheader("🗺️ Heatmap of Ride Demand (Filtered)")
    midpoint = (filtered_df['latitude'].mean(), filtered_df['longitude'].mean())

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=midpoint[0],
            longitude=midpoint[1],
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "HeatmapLayer",
                data=filtered_df,
                get_position="[longitude, latitude]",
                get_weight="rides",
                radiusPixels=40,
                aggregation=pdk.types.String("MEAN"),
            )
        ],
    ))

    # === Auto Insights ===
    st.subheader("🤖 Auto Insights")
    try:
        peak_row = df.loc[df['rides'].idxmax()]
        st.markdown(f"**1. Peak Demand** occurred at `{peak_row['time']}` in location (`{peak_row['latitude']}`, `{peak_row['longitude']}`) with **{peak_row['rides']} rides**.")

        daily_avg = df.groupby(df['time'].dt.date)['rides'].sum().mean()
        st.markdown(f"**2. Average Daily Demand**: `{daily_avg:.2f}` rides per day.")

        busiest_day = df.groupby(df['time'].dt.date)['rides'].sum().idxmax()
        st.markdown(f"**3. Busiest Day**: `{busiest_day}` had the highest total rides.")

    except Exception as e:
        st.error(f"Insight generation failed: {e}")
