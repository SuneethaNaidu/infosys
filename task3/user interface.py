import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction App")

@st.cache_data
def load_data():
    df = pd.read_excel("C:/tasks/archive (3)/UCI_Real_Estate_Valuation.xlsx")

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

X = df.drop("house_price", axis=1)
y = df["house_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

st.sidebar.header("Enter House Details")

transaction_date = st.sidebar.slider(
    "Transaction Date",
    float(df.transaction_date.min()),
    float(df.transaction_date.max()),
    float(df.transaction_date.mean())
)

house_age = st.sidebar.slider("House Age", 0.0, 50.0, 20.0)
distance_to_mrt = st.sidebar.slider("Distance to MRT", 0.0, 6500.0, 1000.0)
num_convenience_stores = st.sidebar.slider("Convenience Stores", 0, 10, 5)
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

input_data = pd.DataFrame([[transaction_date, house_age, distance_to_mrt,
                            num_convenience_stores, latitude, longitude]],
                          columns=X.columns)

prediction = model.predict(input_data)[0]

st.success(f"Predicted House Price: {prediction:.2f}")

st.subheader("House Price Distribution")
fig, ax = plt.subplots()
ax.hist(df["house_price"], bins=30)
ax.set_xlabel("House Price")
ax.set_ylabel("Frequency")
st.pyplot(fig)

st.subheader("Feature Importance")
importance = model.feature_importances_
fig2, ax2 = plt.subplots()
ax2.barh(X.columns, importance)
st.pyplot(fig2)
