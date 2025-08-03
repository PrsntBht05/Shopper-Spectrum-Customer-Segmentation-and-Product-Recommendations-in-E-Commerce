# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# ------------------ Load Data ------------------

@st.cache_data
def load_data():
    return pd.read_csv('c:/Users/bhatt/OneDrive/Desktop/ml project/shop_ml/online_retail.csv')


# ------------------ Load Models ------------------

@st.cache_resource
def load_models():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return kmeans, scaler

# ------------------ Recommendation Engine ------------------

def get_product_recommendations(df, input_product, top_n=5):
    df = df[['CustomerID', 'Description']].dropna()
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['Description'] = df['Description'].str.upper()

    basket = df.pivot_table(index='CustomerID', columns='Description', aggfunc=len, fill_value=0)

    if input_product not in basket.columns:
        return []

    similarity = cosine_similarity(basket.T)
    similarity_df = pd.DataFrame(similarity, index=basket.columns, columns=basket.columns)

    similar_items = similarity_df[input_product].sort_values(ascending=False).iloc[1:top_n+1]
    return similar_items.index.tolist()

# ------------------ Customer Segmentation ------------------

cluster_labels = {
    0: "High-Value",
    1: "Regular",
    2: "Occasional",
    # Add more if applicable
}

def predict_cluster(recency, frequency, monetary):
    kmeans, scaler = load_models()
    input_features = np.array([[recency, frequency, monetary]])
    scaled_features = scaler.transform(input_features)
    cluster = kmeans.predict(scaled_features)[0]
    return cluster_labels.get(cluster, f"Cluster {cluster}")

# ------------------ UI ------------------

def product_recommendation_ui():
    st.header("🛍️ Product Recommendation System")
    df = load_data()
    product_name = st.text_input("📦 Enter Product Name (case-insensitive):").upper()
    if st.button("🔍 Get Recommendations"):
        if product_name:
            recommendations = get_product_recommendations(df, product_name)
            if recommendations:
                st.subheader("🔗 Recommended Products:")
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"**{i}.** {rec}")
            else:
                st.warning("Product not found or not enough data.")
        else:
            st.error("Please enter a valid product name.")

def customer_segmentation_ui():
    st.header("👥 Customer Segmentation")
    st.markdown("Input **Recency**, **Frequency**, and **Monetary** to predict customer segment.")

    recency = st.number_input("📅 Recency (days)", min_value=0, step=1)
    frequency = st.number_input("🔁 Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("💰 Monetary (total spend)", min_value=0.0, step=1.0)

    if st.button("📊 Predict Cluster"):
        label = predict_cluster(recency, frequency, monetary)
        st.success(f"🔖 Predicted Cluster: **{label}**")

# ------------------ Main ------------------

def main():
    st.set_page_config(page_title="E-commerce Intelligence App", layout="centered")
    st.sidebar.title("Navigation")
    module = st.sidebar.radio("Go to Module", ["Product Recommendation", "Customer Segmentation"])

    if module == "Product Recommendation":
        product_recommendation_ui()
    elif module == "Customer Segmentation":
        customer_segmentation_ui()

if __name__ == "__main__":
    main()
