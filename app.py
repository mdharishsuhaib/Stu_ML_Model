from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load the exported pipeline
MODEL_PATH = Path(__file__).resolve().parent / 'final_student_performance_model.joblib'
try:
    model = joblib.load(MODEL_PATH)
    print("Production model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded on the server'}), 500

    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            input_data = [data]
        elif isinstance(data, list):
            input_data = data
        else:
            return jsonify({'error': 'Input data must be a JSON object or a list of objects'}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)

        # Run identical Feature Engineering steps to match the model training schema
        input_df['Study_Efficiency'] = input_df['Hours_Studied'] / (input_df['Sleep_Hours'] + 1)

        support_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        parental_val = input_df['Parental_Involvement'].map(support_mapping).fillna(2)
        resources_val = input_df['Access_to_Resources'].map(support_mapping).fillna(2)
        teacher_val = input_df['Teacher_Quality'].map(support_mapping).fillna(2)
        input_df['Total_Support_Score'] = parental_val + resources_val + teacher_val

        input_df['Academic_Engagement'] = input_df['Hours_Studied'] * (input_df['Tutoring_Sessions'] + 1)

        # Make predictions using the loaded pipeline
        predictions = model.predict(input_df)

        return jsonify({
            'predictions': np.round(predictions, 2).tolist()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
