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
with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    required_fields = ['pregnancies', 'glucose', 'blood_pressure',
                       'skin_thickness', 'insulin', 'bmi', 'dpf', 'age']
    try:
        # Check all fields exist
        for field in required_fields:
            if field not in request.form:
                return render_template('index.html',
                    result=f"Missing field: {field}",
                    result_class="error")

        features = [float(request.form[f]) for f in required_fields]

        # Range checks
        if not (0 <= features[1] <= 300):  # glucose
            return render_template('index.html',
                result="Glucose must be between 0 and 300",
                result_class="error")

        input_array  = np.array([features])
        input_scaled = scaler.transform(input_array)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]

        result       = "⚠️ High Risk of Diabetes" if prediction == 1 else "✅ Low Risk of Diabetes"
        result_class = "danger" if prediction == 1 else "success"
        confidence   = f"{max(probability)*100:.1f}%"

        return render_template('index.html',
            result=result, result_class=result_class, confidence=confidence)

    except ValueError as e:
        return render_template('index.html',
            result="Please enter valid numbers in all fields.",
            result_class="error")
 
import os
if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=5000, debug=debug)
