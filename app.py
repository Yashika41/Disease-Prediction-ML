# ============================================================
# app.py — FLASK WEB APP FOR DISEASE PREDICTION
# Run this file AFTER running disease_prediction.py
# ============================================================

from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the saved model and scaler
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'pipeline.pkl'), 'rb') as f:
    pipeline = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the HTML form
        features = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        # Scale and predict
        input_array  = np.array([features])
        prediction  = pipeline.predict(input_array)[0]
        probability = pipeline.predict_proba(input_array)[0]

        if prediction == 1:
            result      = "⚠️ High Risk of Diabetes"
            result_class = "danger"
            confidence  = f"{probability[1] * 100:.1f}%"
        else:
            result      = "✅ Low Risk of Diabetes"
            result_class = "success"
            confidence  = f"{probability[0] * 100:.1f}%"

        return render_template('index.html',
                               result=result,
                               result_class=result_class,
                               confidence=confidence)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}", result_class="error")


import os
if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=5000, debug=debug)
