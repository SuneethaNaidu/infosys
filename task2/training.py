import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import preprocessing
import joblib
import os

def train_and_evaluate(filepath):
    """
    Trains multiple models and evaluates their performance.
    """
    print(f"Loading data from {filepath}...")
    X, y, preprocessor = preprocessing.load_and_preprocess_data(filepath)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    }
    
    results = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create a pipeline with preprocessor and model
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        
        # Save the best model (for example, Random Forest) or all models
        model_dir = "../models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        joblib.dump(clf, f"{model_dir}/{name.replace(' ', '_').lower()}_model.pkl")
        
    print("\nModel Evaluation Results:")
    results_df = pd.DataFrame(results).T
    print(results_df)

if __name__ == "__main__":
    dataset_path = "../data/world_real_estate_data(147k).csv"
    train_and_evaluate(dataset_path)
