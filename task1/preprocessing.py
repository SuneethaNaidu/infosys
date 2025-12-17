# ==============================
# Weeks 1–2: Data Preparation & EDA
# ==============================
# ==========================================
# Week 1–2: Data Preparation & EDA
# AI-Based Real Estate Valuation System
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------
file_path = "C:/tasks/archive (3)/UCI_Real_Estate_Valuation.xlsx"  # Update path if needed
df = pd.read_excel(file_path)

print("Original Dataset Shape:", df.shape)
print("\nOriginal Columns:\n", df.columns)

# ------------------------------------------
# 2. Drop Unnecessary Column (ID / Index)
# ------------------------------------------
# UCI dataset contains 'No' column which is not useful
if 'No' in df.columns:
    df.drop(columns=['No'], inplace=True)

# ------------------------------------------
# 3. Rename Columns (Safe Method)
# ------------------------------------------
df = df.rename(columns={
    'X1 transaction date': 'transaction_date',
    'X2 house age': 'house_age',
    'X3 distance to the nearest MRT station': 'distance_to_mrt',
    'X4 number of convenience stores': 'num_convenience_stores',
    'X5 latitude': 'latitude',
    'X6 longitude': 'longitude',
    'Y house price of unit area': 'house_price'
})

print("\nRenamed Columns:\n", df.columns)
print("Updated Dataset Shape:", df.shape)

# ------------------------------------------
# 4. Handle Missing Values
# ------------------------------------------
print("\nMissing Values Before Handling:\n", df.isnull().sum())

df.fillna(df.median(numeric_only=True), inplace=True)

print("\nMissing Values After Handling:\n", df.isnull().sum())

# ------------------------------------------
# 5. Handle Outliers using IQR Method
# ------------------------------------------
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df = remove_outliers(df, col)

print("\nDataset Shape After Outlier Removal:", df.shape)

# ------------------------------------------
# 6. Check House Price Distribution (Balance)
# ------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(df['house_price'], bins=30, kde=True)
plt.title("House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.show()

# ------------------------------------------
# 7. Exploratory Data Analysis (EDA)
# ------------------------------------------

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pair Plot (Relationships)
sns.pairplot(df)
plt.show()

# ------------------------------------------
# 8. Summary Statistics
# ------------------------------------------
print("\nSummary Statistics:\n")
print(df.describe())

print("\nWeek 1–2 Data Preparation & EDA Completed Successfully ✅")




