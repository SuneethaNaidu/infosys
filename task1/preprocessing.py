# ==============================
# Weeks 1–2: Data Preparation & EDA
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
# Update the path if needed
file_path = "C:\tasks\archive (3)\UCI_Real_Estate_Valuation.xlsx"
df = pd.read_excel(file_path)

print("Dataset Shape:", df.shape)
print(df.head())

# 2. Rename columns (optional but recommended for clarity)
df.columns = [
    "transaction_date",
    "house_age",
    "distance_to_mrt",
    "num_convenience_stores",
    "latitude",
    "longitude",
    "house_price"
]

# 3. Handle Missing Values
print("\nMissing values before handling:\n", df.isnull().sum())

df.fillna(df.median(numeric_only=True), inplace=True)

print("\nMissing values after handling:\n", df.isnull().sum())

# 4. Handle Outliers using IQR
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

for col in df.select_dtypes(include=np.number).columns:
    df = remove_outliers(df, col)

print("\nDataset shape after outlier removal:", df.shape)

# 5. Check Price Distribution (Balanced Price Ranges)
plt.figure(figsize=(6,4))
sns.histplot(df["house_price"], bins=30, kde=True)
plt.title("House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.show()

# 6. Exploratory Data Analysis (EDA)
# Correlation Matrix
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plots
sns.pairplot(df)
plt.show()

