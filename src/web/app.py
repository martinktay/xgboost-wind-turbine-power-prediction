from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# Load the latest model and scaler


def load_latest_model():
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.startswith(
        'xgboost_model_') and f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    return joblib.load(os.path.join(models_dir, latest_model))


def load_latest_metrics():
    models_dir = 'models'
    metrics_files = [f for f in os.listdir(models_dir) if f.startswith(
        'model_metrics_') and f.endswith('.json')]
    if not metrics_files:
        return None
    latest_metrics = max(metrics_files)
    with open(os.path.join(models_dir, latest_metrics), 'r') as f:
        return json.load(f)


# Load the model and scaler at startup
try:
    model = load_latest_model()
    scaler = joblib.load('models/scaler.pkl')
    metrics = load_latest_metrics()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    scaler = None
    metrics = None


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', metrics=metrics)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    try:
        data = request.json
        print("\n=== Request Data ===")
        print("Raw data:", data)

        # Add historical features (using current ActivePower as a proxy)
        current_power = float(data['ActivePower'])  # Ensure numeric type
        data['lag_1h_ActivePower'] = current_power
        data['lag_24h_ActivePower'] = current_power
        data['lag_7d_ActivePower'] = current_power
        data['rolling_mean_24h_ActivePower'] = current_power
        data['rolling_mean_7d_ActivePower'] = current_power

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([data])
        print("\n=== Input DataFrame ===")
        print("Columns:", input_data.columns.tolist())
        print("Data types:", input_data.dtypes)

        # Convert all numeric columns to float64
        for col in input_data.columns:
            try:
                input_data[col] = pd.to_numeric(
                    input_data[col], errors='raise')
            except Exception as e:
                print(f"Error converting {col} to numeric:", str(e))

        print("\n=== After Numeric Conversion ===")
        print("Data types:", input_data.dtypes)

        # Scale using all features (including ActivePower)
        try:
            # Get scaler features and ensure they're in the correct order
            scaling_features = scaler.feature_names_in_.tolist()
            print("\n=== Scaling Features ===")
            print("Expected features:", scaling_features)
            print("Available features:", input_data.columns.tolist())

            # Check for missing features
            missing_features = set(scaling_features) - set(input_data.columns)
            if missing_features:
                return jsonify({
                    'error': 'Missing features',
                    'message': f'Missing required features: {missing_features}',
                    'missing_features': list(missing_features)
                }), 400

            # Reorder columns to match scaler's expected order
            scaling_data = input_data[scaling_features].copy()
            print("\n=== Scaling Data ===")
            print("Shape:", scaling_data.shape)
            print("Features:", scaling_data.columns.tolist())

            # Scale the data
            scaled_data = scaler.transform(scaling_data)
            scaled_df = pd.DataFrame(scaled_data, columns=scaling_features)

            # Get prediction features (excluding ActivePower)
            prediction_features = model.feature_names_in_.tolist()
            prediction_data = scaled_df[prediction_features]
            print("\n=== Prediction Data ===")
            print("Shape:", prediction_data.shape)
            print("Features:", prediction_data.columns.tolist())

        except Exception as e:
            print("\n=== Scaling Error ===")
            print("Error message:", str(e))
            return jsonify({
                'error': 'Scaling error',
                'message': str(e),
                'details': {
                    'input_shape': list(input_data.shape),
                    'input_features': input_data.columns.tolist(),
                    'expected_features': scaling_features
                }
            }), 400

        # Make prediction
        try:
            prediction = model.predict(prediction_data)
            print("\n=== Prediction Success ===")
            print("Prediction value:", prediction[0])
            return jsonify({
                'prediction': float(prediction[0]),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        except Exception as e:
            print("\n=== Prediction Error ===")
            print("Error message:", str(e))
            return jsonify({
                'error': 'Prediction error',
                'message': str(e)
            }), 400

    except Exception as e:
        print("\n=== Unexpected Error ===")
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 400


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch predictions"""
    try:
        data = request.json

        # Create a DataFrame from the input data
        input_data = pd.DataFrame(data['features'])

        # Scale the input features
        scaled_data = scaler.transform(input_data)

        # Make predictions
        predictions = model.predict(scaled_data)

        return jsonify({
            'predictions': predictions.tolist(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/model_info')
def model_info():
    """Return model performance metrics"""
    if metrics is None:
        return jsonify({'error': 'No model metrics available'}), 404
    return jsonify(metrics)


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
