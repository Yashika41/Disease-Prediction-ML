# ============================================================
# app.py — FIXED VERSION
# Changes: pipeline.pkl instead of model.pkl + scaler.pkl
#          Fixed debug mode
#          Fixed duplicate pickle load
#          Better input validation
# ============================================================

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the saved pipeline (one file instead of two!)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'pipeline.pkl'), 'rb') as f:
    pipeline = pickle.load(f)

# Feature names (same as training data columns)
FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure',
                 'SkinThickness', 'Insulin', 'BMI',
                 'DiabetesPedigreeFunction', 'Age']

@app.route('/health')
def health():
    return {'status': 'ok', 'model': 'loaded'}, 200
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Field names from the HTML form
    form_fields = ['pregnancies', 'glucose', 'blood_pressure',
                   'skin_thickness', 'insulin', 'bmi', 'dpf', 'age']
    try:
        # Check all fields exist
        for field in form_fields:
            if field not in request.form or request.form[field] == '':
                return render_template('index.html',
                    result=f"⚠️ Please fill in all fields.",
                    result_class="error",form_data=request.form)

        # Get values from form
        values = [float(request.form[f]) for f in form_fields]

        # Basic range checks
        glucose = values[1]
        bmi     = values[5]
        age     = values[7]

        if not (0 <= glucose <= 300):
            return render_template('index.html',
                result="⚠️ Glucose must be between 0 and 300",
                result_class="error",form_data=request.form)
        if not (0 <= bmi <= 70):
            return render_template('index.html',
                result="⚠️ BMI must be between 0 and 70",
                result_class="error",form_data=request.form)
        if not (1 <= age <= 120):
            return render_template('index.html',
                result="⚠️ Age must be between 1 and 120",
                result_class="error",form_data=request.form)

        # Create DataFrame with correct column names (fixes the warning!)
        input_df = pd.DataFrame([values], columns=FEATURE_NAMES)

        # Predict using pipeline (handles scaling automatically!)
        prediction  = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0]

        if prediction == 1:
            result       = "⚠️ High Risk of Diabetes"
            result_class = "danger"
        else:
            result       = "✅ Low Risk of Diabetes"
            result_class = "success"

        confidence = f"{max(probability)*100:.1f}%"

        return render_template('index.html',
                       result=result,
                       result_class=result_class,
                       confidence=confidence,
                       form_data={
                           'pregnancies':    request.form.get('pregnancies', ''),
                           'glucose':        request.form.get('glucose', ''),
                           'blood_pressure': request.form.get('blood_pressure', ''),
                           'skin_thickness': request.form.get('skin_thickness', ''),
                           'insulin':        request.form.get('insulin', ''),
                           'bmi':            request.form.get('bmi', ''),
                           'dpf':            request.form.get('dpf', ''),
                           'age':            request.form.get('age', '')
                       })

    except ValueError:
        return render_template('index.html',
    result=f"⚠️ Please fill in all fields.",
    result_class="error",
    form_data=request.form)

if __name__ == '__main__':
    # Safe debug mode — reads from environment variable
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=5000, debug=debug)