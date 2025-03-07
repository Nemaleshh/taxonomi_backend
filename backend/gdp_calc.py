from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Define the transformer class FIRST
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
            # Load model info first
            with open("models/model_info.json", "r") as f:
                MODEL_INFO = json.load(f)
            
            # Load model with custom transformer
            MODEL = joblib.load(MODEL_INFO["model_path"])
            print(f"Successfully loaded {MODEL_INFO['best_model']} model")
            
        return True
    except Exception as e:
        print(f"Model load error: {str(e)}")
        try:
            # Try loading fallback model
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
        # Validate and extract input values
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

        # Create input array in correct order
        input_array = np.array([[input_values[feat] for feat in REQUIRED_FEATURES]])
        
        # Make prediction
        prediction = (MODEL.predict(input_array)[0])/2
        
        # Validate prediction
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

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    if MODEL is None and not load_model():
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_type": MODEL_INFO.get("best_model", "unknown"),
        "training_date": MODEL_INFO.get("creation_date", "unknown"),
        "r2_score": MODEL_INFO.get("best_score", "unknown")
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "model_loaded": MODEL is not None
    })

if __name__ == '__main__':
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Load model before starting server
    load_model()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)