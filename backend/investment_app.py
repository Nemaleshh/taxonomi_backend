from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# Load artifacts
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

def calculate_allocations(total_budget):
    try:
        # Create total calculated column if missing
        if 'TOTAL_CALCULATED' not in processed_data.columns:
            processed_data['TOTAL_CALCULATED'] = processed_data[main_sectors].sum(axis=1)

        allocations = {}
        # Main allocation calculation
        for sector in main_sectors:
            if sector in trend_models:
                model = trend_models[sector]
                latest_year = processed_data['Year'].max()
                pred_year = latest_year + 1
                predicted = model.predict([[pred_year]])[0]
                allocations[sector] = max(0.04 * total_budget, predicted)

        # Normalize allocations
        total = sum(allocations.values())
        for sector in allocations:
            allocations[sector] = (allocations[sector] / total) * total_budget

        # Historical averages
        historical_avg = {
            sector: processed_data[sector].mean() / processed_data['TOTAL_CALCULATED'].mean() * 100
            for sector in main_sectors
        }

        # Recent trends (last 5 years)
        recent_data = processed_data.nlargest(5, 'Year')
        recent_avg = {
            sector: recent_data[sector].mean() / recent_data['TOTAL_CALCULATED'].mean() * 100
            for sector in main_sectors
        }

        # Growth rates
        growth_rates = {}
        for sector in main_sectors:
            y_values = processed_data[sector].values
            if len(y_values) > 1:
                growth = (y_values[-1] / y_values[0]) ** (1/len(y_values)) - 1
                growth_rates[sector] = growth * 100

        # Insights
        insights = []
        for sector in main_sectors:
            if recent_avg[sector] > historical_avg[sector]:
                insights.append(f"{sector} spending has been increasing in recent years.")
            else:
                insights.append(f"{sector} spending has been decreasing in recent years.")

        # Subsector allocations (example)
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

def preprocess_data(data):
    # Convert to numeric, coerce errors to NaN
    for col in data.columns:
        if col != 'Year':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Fill NaN values with column means
    data = data.fillna(data.mean())
    
    # Create total calculated column
    main_sectors = [
        'A. NON-DEVELOPMENTAL EXPENDITURE',
        'B. DEVELOPMENTAL EXPENDITURE',
        'C. LOANS AND ADVANCES'
    ]
    data['TOTAL_CALCULATED'] = data[main_sectors].sum(axis=1)
    
    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)