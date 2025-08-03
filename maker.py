import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# Load raw data
df = pd.read_csv("c:/Users/bhatt/OneDrive/Desktop/ml project/shop_ml/online_retail.csv", encoding='ISO-8859-1')

# Clean up missing data
df.dropna(subset=['CustomerID'], inplace=True)
df['CustomerID'] = df['CustomerID'].astype(str)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ✅ Add this to compute total spend per row
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Reference date
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Build RFM table
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
})

# Remove customers with negative/zero spend
rfm = rfm[rfm['Monetary'] > 0]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm)

# Train
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Save
with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ KMeans model and scaler saved successfully.")
