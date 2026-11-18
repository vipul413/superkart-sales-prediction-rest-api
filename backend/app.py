import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask Application
superkart_api = Flask("SuperKart Sales Predictor")

# Load serialized model pipeline
model = joblib.load("superkart_model.joblib")

@superkart_api.get('/')
def home():
    """Root route confirming API availability."""
    return "Welcome to the SuperKart Sales Prediction API!"

@superkart_api.post('/v1/predict')
def predict():
    """Online (single item) prediction endpoint accepting JSON."""
    try:
        data = request.get_json(force=True)
        input_data = pd.DataFrame([data])
        
        prediction = model.predict(input_data)[0]
        prediction = round(float(prediction), 2)
        
        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@superkart_api.post('/v1/predictbatch')
def predict_batch():
    """Batch prediction endpoint accepting a CSV file."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded under key 'file'"}), 400
            
        file = request.files['file']
        input_data = pd.read_csv(file)
        
        predictions = model.predict(input_data).tolist()
        predicted_sales = [round(float(p), 2) for p in predictions]
        
        # Return index: prediction dictionary
        output_dict = {str(idx): val for idx, val in enumerate(predicted_sales)}
        return jsonify(output_dict), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    superkart_api.run(host="0.0.0.0", port=7860, debug=True)
