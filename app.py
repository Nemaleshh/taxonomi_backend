from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# --- GDP Prediction Code ---
class EconomicFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_interaction=True):
        self.add_interaction = add_interaction
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[
                'unemployment_rate', 'personal_consumption',
                'govt_expenditure', 'm1_money_supply',
                'm2_money_supply', 'federal_debt'
            ])
        else:
            X_df = X.copy()
            
        X_scaled = self.scaler.transform(X_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)
        result = X_scaled_df.copy()
        
        if self.add_interaction:
            result['consumption_ratio'] = X_scaled_df['personal_consumption'] / (
                X_scaled_df['personal_consumption'] + X_scaled_df['govt_expenditure']
            )
            result['m2_m1_ratio'] = X_scaled_df['m2_money_supply'] / X_scaled_df['m1_money_supply'].replace(0, 0.001)
            result['debt_spending_ratio'] = X_scaled_df['federal_debt'] / (
                X_scaled_df['personal_consumption'] + X_scaled_df['govt_expenditure']
            ).replace(0, 0.001)
            result['unemployment_consumption'] = X_scaled_df['unemployment_rate'] * X_scaled_df['personal_consumption']
            result['log_consumption'] = np.log1p(np.abs(X_scaled_df['personal_consumption']))
            result['log_govt_exp'] = np.log1p(np.abs(X_scaled_df['govt_expenditure']))
        
        return result.values

MODEL = None
MODEL_INFO = None
REQUIRED_FEATURES = [
    'unemployment_rate',
    'personal_consumption',
    'govt_expenditure',
    'm1_money_supply',
    'm2_money_supply',
    'federal_debt'
]

def load_model():
    global MODEL, MODEL_INFO
    try:
        if MODEL is None:
            with open("models/model_info.json", "r") as f:
                MODEL_INFO = json.load(f)
            MODEL = joblib.load(MODEL_INFO["model_path"])
            print(f"Successfully loaded {MODEL_INFO['best_model']} model")
        return True
    except Exception as e:
        print(f"Model load error: {str(e)}")
        try:
            MODEL = joblib.load("models/fallback_gdp_model.pkl")
            MODEL_INFO = {
                "best_model": "fallback",
                "features": REQUIRED_FEATURES
            }
            print("Loaded fallback model")
            return True
        except Exception as fallback_error:
            print(f"Fallback load failed: {str(fallback_error)}")
            return False

@app.route('/api/predict', methods=['POST'])
def predict_gdp():
    if not load_model():
        return jsonify({
            "error": "Model Error",
            "message": "Could not load any prediction model"
        }), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        input_values = {}
        for feature in REQUIRED_FEATURES:
            value = data.get(feature)
            if value is None:
                return jsonify({
                    "error": "Missing field",
                    "message": f"{feature} is required"
                }), 400
            
            try:
                num_value = float(value)
                if feature == 'unemployment_rate' and not (0 <= num_value <= 100):
                    return jsonify({
                        "error": "Invalid value",
                        "message": "Unemployment rate must be between 0-100"
                    }), 400
                input_values[feature] = num_value
            except ValueError:
                return jsonify({
                    "error": "Invalid value",
                    "message": f"{feature} must be a number"
                }), 400

        input_array = np.array([[input_values[feat] for feat in REQUIRED_FEATURES]])
        prediction = (MODEL.predict(input_array)[0])/2
        
        if not np.isfinite(prediction):
            prediction = (input_values['personal_consumption'] * 0.5) + (input_values['govt_expenditure'] * 0.3)
        
        return jsonify({
            "gdp_prediction": float(prediction),
            "currency": "₹ Crores",
            "model_used": MODEL_INFO.get("best_model", "unknown"),
            "input_values": input_values
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            "error": "Prediction Failed",
            "message": str(e)
        }), 500

# --- Investment Allocation Code ---
main_sectors = None
trend_models = None
processed_data = None
_initialized = False

def load_artifacts():
    global main_sectors, trend_models, processed_data
    try:
        main_sectors = joblib.load('investment_models/main_sectors.pkl')
        trend_models = joblib.load('investment_models/trend_models.pkl')
        processed_data = pd.read_csv('investment_models/processed_data.csv')
        print("Artifacts loaded successfully")
    except Exception as e:
        print(f"Error loading artifacts: {str(e)}")
        raise

@app.before_request
def initialize():
    global _initialized
    if not _initialized:
        try:
            print("Initializing application...")
            os.makedirs('investment_models', exist_ok=True)
            load_artifacts()
            _initialized = True
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            raise

@app.route('/api/allocate', methods=['POST'])
def allocate():
    try:
        data = request.get_json()
        budget = float(data['budget'])
        
        if budget < 5000:
            return jsonify({'error': 'Minimum budget must be ₹5000 crores'}), 400
        
        result = calculate_allocations(budget)
        if not result:
            return jsonify({'error': 'Failed to calculate allocations'}), 500
            
        return jsonify({
            'main_allocations': result['allocations'],
            'historical': result['historical_avg'],
            'recent_trends': result['recent_avg'],
            'growth_rates': result['growth_rates'],
            'insights': result['insights'],
            'subsectors': result['subsectors'],
            'total_budget': budget
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_allocations(total_budget):
    try:
        if 'TOTAL_CALCULATED' not in processed_data.columns:
            processed_data['TOTAL_CALCULATED'] = processed_data[main_sectors].sum(axis=1)

        allocations = {}
        for sector in main_sectors:
            if sector in trend_models:
                model = trend_models[sector]
                latest_year = processed_data['Year'].max()
                pred_year = latest_year + 1
                predicted = model.predict([[pred_year]])[0]
                allocations[sector] = max(0.04 * total_budget, predicted)

        total = sum(allocations.values())
        for sector in allocations:
            allocations[sector] = (allocations[sector] / total) * total_budget

        historical_avg = {
            sector: processed_data[sector].mean() / processed_data['TOTAL_CALCULATED'].mean() * 100
            for sector in main_sectors
        }

        recent_data = processed_data.nlargest(5, 'Year')
        recent_avg = {
            sector: recent_data[sector].mean() / recent_data['TOTAL_CALCULATED'].mean() * 100
            for sector in main_sectors
        }

        growth_rates = {}
        for sector in main_sectors:
            y_values = processed_data[sector].values
            if len(y_values) > 1:
                growth = (y_values[-1] / y_values[0]) ** (1/len(y_values)) - 1
                growth_rates[sector] = growth * 100

        insights = []
        for sector in main_sectors:
            if recent_avg[sector] > historical_avg[sector]:
                insights.append(f"{sector} spending has been increasing in recent years.")
            else:
                insights.append(f"{sector} spending has been decreasing in recent years.")

        subsectors = {
            '8. Transport & Communication': 0.2265 * total_budget,
            '7. Power, irrigation & flood control': 0.2171 * total_budget,
            '3. Social and Community Services': 0.2037 * total_budget
        }

        return {
            'allocations': allocations,
            'historical_avg': historical_avg,
            'recent_avg': recent_avg,
            'growth_rates': growth_rates,
            'insights': insights,
            'subsectors': subsectors
        }
    
    except Exception as e:
        print(f"Calculation error: {str(e)}")
        return None

if __name__ == '__main__':

    load_model()
    load_artifacts()
    app.run()