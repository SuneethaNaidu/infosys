import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction App")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_excel("C:/tasks/archive (3)/UCI_Real_Estate_Valuation.xlsx" )

    if "No" in df.columns:
        df.drop(columns=["No"], inplace=True)

    df.rename(columns={
        "X1 transaction date": "transaction_date",
        "X2 house age": "house_age",
        "X3 distance to the nearest MRT station": "distance_to_mrt",
        "X4 number of convenience stores": "num_convenience_stores",
        "X5 latitude": "latitude",
        "X6 longitude": "longitude",
        "Y house price of unit area": "house_price"
    }, inplace=True)

    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

df = load_data()

X = df.drop("house_price", axis=1)
y = df["house_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

st.sidebar.header("Enter House Details")

inputs = {}
for col in X.columns:
    inputs[col] = st.sidebar.number_input(
        col.replace("_", " ").title(),
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )

input_df = pd.DataFrame([inputs])
prediction = model.predict(input_df)[0]

st.success(f"💰 Predicted House Price: {prediction:.2f}")

st.subheader("📊 House Price Distribution")
fig, ax = plt.subplots()
ax.hist(df["house_price"], bins=30)
ax.set_xlabel("House Price")
ax.set_ylabel("Frequency")
st.pyplot(fig)

st.subheader("📌 Feature Importance")
fig2, ax2 = plt.subplots()
ax2.barh(X.columns, model.feature_importances_)
st.pyplot(fig2)

