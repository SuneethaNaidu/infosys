\import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction App",
    layout="wide"
)

st.title("🏠 House Price Prediction & Market Analysis")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    file_path = "C:/tasks/archive (3)/UCI_Real_Estate_Valuation.xlsx" 
    df = pd.read_excel(file_path)

    if 'No' in df.columns:
        df.drop(columns=['No'], inplace=True)

    df = df.rename(columns={
        'X1 transaction date': 'transaction_date',
        'X2 house age': 'house_age',
        'X3 distance to the nearest MRT station': 'distance_to_mrt',
        'X4 number of convenience stores': 'num_convenience_stores',
        'X5 latitude': 'latitude',
        'X6 longitude': 'longitude',
        'Y house price of unit area': 'house_price'
    })

    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

df = load_data()

# -----------------------------
# MODEL TRAINING
# -----------------------------
X = df.drop("house_price", axis=1)
y = df["house_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# SIDEBAR - USER INPUT
# -----------------------------
st.sidebar.header("📋 Enter House Details")

transaction_date = st.sidebar.slider(
    "Transaction Date",
    float(df.transaction_date.min()),
    float(df.transaction_date.max()),
    float(df.transaction_date.mean())
)

house_age = st.sidebar.slider(
    "House Age (years)",
    0.0, 50.0, 20.0
)

distance_to_mrt = st.sidebar.slider(
    "Distance to MRT (meters)",
    0.0, 6500.0, 1000.0
)

num_convenience_stores = st.sidebar.slider(
    "Number of Convenience Stores",
    0, 10, 5
)

latitude = st.sidebar.slider(
    "Latitude",
    float(df.latitude.min()),
    float(df.latitude.max()),
    float(df.latitude.mean())
)

longitude = st.sidebar.slider(
    "Longitude",
    float(df.longitude.min()),
    float(df.longitude.max()),
    float(df.longitude.mean())
)

# -----------------------------
# PREDICTION
# -----------------------------
input_data = pd.DataFrame([[
    transaction_date,
    house_age,
    distance_to_mrt,
    num_convenience_stores,
    latitude,
    longitude
]], columns=X.columns)

prediction = model.predict(input_data)[0]

st.subheader("💰 Predicted House Price")
st.success(f"Estimated Price per Unit Area: **{prediction:.2f}**")

# -----------------------------
# VISUALIZATIONS
# -----------------------------
st.markdown("---")
st.subheader("📊 Data Insights & Visualizations")

col1, col2 = st.columns(2)

# ---- Price Distribution
with col1:
    st.markdown("### House Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["house_price"], bins=30, kde=True, ax=ax1)
    ax1.set_xlabel("House Price")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

# ---- Feature Importance
with col2:
    st.markdown("### Feature Importance")
    importance = model.feature_importances_
    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots()
    sns.barplot(
        data=feature_df,
        x="Importance",
        y="Feature",
        ax=ax2
    )
    st.pyplot(fig2)

# -----------------------------
# MARKET TRENDS
# -----------------------------
st.markdown("---")
st.subheader("📈 Market Trend Analysis")

fig3, ax3 = plt.subplots()
sns.scatterplot(
    data=df,
    x="house_age",
    y="house_price",
    ax=ax3
)
ax3.set_xlabel("House Age (Years)")
ax3.set_ylabel("House Price")
ax3.set_title("House Price vs House Age")
st.pyplot(fig3)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
    ---
    ✅ **Model Used:** Random Forest Regressor  
    📊 **Dataset:** UCI Real Estate Valuation  
    🚀 Built with Streamlit
    """
)
