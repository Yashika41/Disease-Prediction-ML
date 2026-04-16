# ============================================================
# app.py — FLASK WEB APP FOR DISEASE PREDICTION
# Run this file AFTER running disease_prediction.py
# ============================================================

from flask import Flask, render_template, request
import numpy as np
import pickle
import os
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

app = Flask(__name__)

# Load the saved model and scaler
model  = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

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
        input_scaled = scaler.transform(input_array)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]

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


if __name__ == '__main__':
    app.run(debug=True)
