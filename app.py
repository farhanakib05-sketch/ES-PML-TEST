import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load saved artifacts on startup
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.joblib'

model = None
scaler = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Successfully loaded model.pkl and scaler.joblib on startup.")
    else:
        print("Warning: model.pkl or scaler.joblib not found in directory.")
except Exception as e:
    print(f"Error loading artifacts on startup: {str(e)}")

# Feature configuration matching training dummy columns
NUMERICAL_FEATURES = ['Age', 'Annual_Income', 'Years_Experience']
FEATURE_COLUMNS = [
    'Age', 'Annual_Income', 'Years_Experience',
    'Education_Level_High School', 'Education_Level_Master', 'Education_Level_Phd',
    'City_Houston', 'City_Los Angeles', 'City_New York', 'City_Phoenix'
]
VALID_EDUCATION_LEVELS = ['High School', 'Bachelor', 'Master', 'PhD', 'Phd']
VALID_CITIES = ['Chicago', 'Houston', 'Los Angeles', 'New York', 'Phoenix']


@app.route('/')
def index():
    """GET route: Serves HTML interface"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """POST route: Accepts JSON payload, validates, scales, predicts, returns JSON response"""
    if model is None or scaler is None:
        return jsonify({
            'success': False,
            'error': 'Model or scaler artifact not loaded on server.'
        }), 500

    if not request.is_json:
        return jsonify({
            'success': False,
            'error': 'Request Content-Type must be application/json.'
        }), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            'success': False,
            'error': 'Invalid or empty JSON payload provided.'
        }), 400

    # Validate required input fields
    required_fields = ['Age', 'Annual_Income', 'Years_Experience', 'Education_Level', 'City']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({
            'success': False,
            'error': f'Missing required input fields: {", ".join(missing_fields)}'
        }), 400

    # Validate data types and value bounds
    try:
        age = float(data['Age'])
        income = float(data['Annual_Income'])
        experience = float(data['Years_Experience'])
        education = str(data['Education_Level']).strip().title()
        city = str(data['City']).strip()
    except (ValueError, TypeError):
        return jsonify({
            'success': False,
            'error': 'Age, Annual_Income, and Years_Experience must be valid numeric values.'
        }), 400

    if age < 0 or age > 120:
        return jsonify({'success': False, 'error': 'Age must be between 0 and 120.'}), 400
    if income < 0:
        return jsonify({'success': False, 'error': 'Annual_Income cannot be negative.'}), 400
    if experience < 0 or experience > 80:
        return jsonify({'success': False, 'error': 'Years_Experience must be between 0 and 80.'}), 400

    if education not in [e.title() for e in VALID_EDUCATION_LEVELS]:
        return jsonify({
            'success': False,
            'error': f'Invalid Education_Level. Must be one of: {", ".join(VALID_EDUCATION_LEVELS)}'
        }), 400

    if city not in VALID_CITIES:
        return jsonify({
            'success': False,
            'error': f'Invalid City. Must be one of: {", ".join(VALID_CITIES)}'
        }), 400

    # Build raw input dataframe
    raw_df = pd.DataFrame([{
        'Age': age,
        'Annual_Income': income,
        'Years_Experience': experience,
        'Education_Level': education,
        'City': city
    }])

    # One-hot encoding & column alignment matching model training schema
    encoded_df = pd.get_dummies(raw_df, columns=['Education_Level', 'City'], drop_first=True, dtype=float)
    aligned_df = encoded_df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)

    # Scale numerical features using fitted StandardScaler
    scaled_df = aligned_df.copy()
    scaled_df[NUMERICAL_FEATURES] = scaler.transform(aligned_df[NUMERICAL_FEATURES])

    # Model inference
    try:
        prediction_val = float(model.predict(scaled_df)[0])
        return jsonify({
            'success': True,
            'prediction': round(prediction_val, 2),
            'formatted_prediction': f"${prediction_val:,.2f}"
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Model inference error: {str(e)}'
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found.'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
