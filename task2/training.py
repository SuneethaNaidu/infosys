import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
file_path ="C:/tasks/archive (3)/UCI_Real_Estate_Valuation.xlsx"
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
X = df.drop('house_price', axis=1)
y = df['house_price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        objective="reg:squarederror"
    )
}
results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append([model_name, rmse, mae, r2])
results_df = pd.DataFrame(
    results, columns=["Model", "RMSE", "MAE", "R² Score"]
)
print("\nModel Performance Comparison:\n")
print(results_df)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print("\nBest Random Forest Parameters:")
print(grid_search.best_params_)
y_pred_best = best_rf.predict(X_test)
print("\nOptimized Random Forest Performance:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_best)))
print("MAE :", mean_absolute_error(y_test, y_pred_best))
print("R²  :", r2_score(y_test, y_pred_best))





