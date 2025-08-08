import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Ahmedabad coordinates
lat_center, lon_center = 23.0225, 72.5714

# Generate 1000 random latitude and longitude points around the center
latitudes = np.random.normal(loc=lat_center, scale=0.01, size=1000)
longitudes = np.random.normal(loc=lon_center, scale=0.01, size=1000)

# Generate random timestamps between Aug 6 and Aug 21, 2025
date_range = pd.date_range(start="2025-08-06", end="2025-08-21", freq="H")
timestamps = np.random.choice(date_range, size=1000)

# Create the DataFrame
df = pd.DataFrame({
    "latitude": latitudes,
    "longitude": longitudes,
    "time": timestamps
})

# Save to CSV
df.to_csv("ahmedabad_demand_aug6_to_aug21.csv", index=False)

print("✅ CSV generated successfully!")
