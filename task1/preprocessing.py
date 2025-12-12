import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def clean_area_column(column):
    """Removes ' m²' and converts to float."""
    if column.dtype == object:
        return column.str.replace(' m²', '', regex=False).str.replace(',', '', regex=False).astype(float)
    return column

def load_and_preprocess_data(filepath):
    """
    Loads the dataset, performs cleaning, and prepares features/target.
    """
    df = pd.read_csv(filepath)
    
    # Target variable
    target_col = 'price_in_USD'
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])
    
    # Clean area columns
    area_cols = ['apartment_total_area', 'apartment_living_area']
    for col in area_cols:
        if col in df.columns:
            df[col] = clean_area_column(df[col])
            
    # Select relevant features
    # Numerical features
    numeric_features = [
        'building_construction_year', 'building_total_floors', 
        'apartment_floor', 'apartment_rooms', 'apartment_bedrooms', 
        'apartment_bathrooms', 'apartment_total_area', 'apartment_living_area'
    ]
    
    # Categorical features
    categorical_features = ['country', 'location']
    
    # Filter features that exist in the dataframe
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    X = df[numeric_features + categorical_features]
    y = df[target_col]
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for dense matrix if needed, or True for memory efficiency. XGBoost handles sparse.
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

if __name__ == "__main__":
    # Test the loading function
    try:
        X, y, preprocessor = load_and_preprocess_data('../data/world_real_estate_data(147k).csv')
        print("Data loaded successfully.")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
