import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression

def save_model_artifacts():
    # Load and preprocess data
    data = pd.read_csv("backend\\merged_expenditure_dataset.csv")
    data = preprocess_data(data)
    
    # Train and save trend models for each sector
    main_sectors = [
        'A. NON-DEVELOPMENTAL EXPENDITURE',
        'B. DEVELOPMENTAL EXPENDITURE',
        'C. LOANS AND ADVANCES'
    ]
    
    trend_models = {}
    for sector in main_sectors:
        sector_data = data[['Year', sector]].dropna()
        if len(sector_data) > 1:
            X = sector_data['Year'].values.reshape(-1, 1)
            y = sector_data[sector].values
            model = LinearRegression()
            model.fit(X, y)
            trend_models[sector] = model
    
    # Save artifacts
    os.makedirs('investment_models', exist_ok=True)
    joblib.dump(trend_models, 'investment_models/trend_models.pkl')
    joblib.dump(main_sectors, 'investment_models/main_sectors.pkl')
    data.to_csv('investment_models/processed_data.csv', index=False)

def preprocess_data(data):
    # Convert to numeric, coerce errors to NaN
    for col in data.columns:
        if col != 'Year':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Fill NaN values with column means
    data = data.fillna(data.mean())
    return data

if __name__ == '__main__':
    save_model_artifacts()