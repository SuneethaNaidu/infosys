import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Load model & data
# ---------------------------
model = joblib.load("house_price_model.pkl")
data = pd.read_excel("C:/tasks/archive (3)/UCI_Real_Estate_Valuation.xlsx")

# ---------------------------
# App title
# ---------------------------
st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction System")

st.markdown("Enter house details to predict the **estimated price** and explore market insights.")

# ---------------------------
# Sidebar – User Inputs
# ---------------------------
st.sidebar.header("🔧 House Features")

area = st.sidebar.number_input("Area (sq.ft)", 500, 5000, 1200)
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 4, 2)
stories = st.sidebar.slider("Stories", 1, 4, 1)
parking = st.sidebar.slider("Parking Spaces", 0, 3, 1)

# Convert inputs to DataFrame
input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, parking]],
                          columns=["area", "bedrooms", "bathrooms", "stories", "parking"])

# ---------------------------
# Prediction
# ---------------------------
if st.sidebar.button("🔮 Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated House Price: ₹ {prediction:,.2f}")

# ---------------------------
# Layout for visualizations
# ---------------------------
col1, col2 = st.columns(2)

# ---------------------------
# Price Distribution
# ---------------------------
with col1:
    st.subheader("📊 Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data["price"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("House Price")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# ---------------------------
# Market Trend (Area vs Price)
# ---------------------------
with col2:
    st.subheader("📈 Market Trend (Area vs Price)")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data["area"], y=data["price"], ax=ax)
    ax.set_xlabel("Area (sq.ft)")
    ax.set_ylabel("Price")
    st.pyplot(fig)

# ---------------------------
# Feature Importance
# ---------------------------
st.subheader("⭐ Feature Importance")

if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_
    features = input_data.columns

    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=features, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)
else:
    st.info("Feature importance not available for this model.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("📌 **Note:** This prediction is based on historical data and machine learning models.")
