# Model Card — Diabetes Risk Predictor

## Model Details
- **Type:** Random Forest Classifier (sklearn Pipeline)
- **Version:** 1.0
- **Training Date:** May 2026
- **Developer:** Yashika Sharma — github.com/Yashika41

## Intended Use
- ✅ Educational demonstration of ML for health risk assessment
- ✅ Portfolio project showing end-to-end ML pipeline
- ❌ NOT for clinical or diagnostic use
- ❌ NOT a replacement for professional medical advice

## Training Data
- **Dataset:** Pima Indians Diabetes Database
- **Source:** UCI ML Repository / Kaggle
- **Size:** 768 samples, 9 features
- **Population:** Female Pima Indian patients, aged 21+

## ⚠️ Known Limitations & Bias
- Trained ONLY on **female Pima Indian patients aged 21+**
- Will perform **worse** on: males, children, non-Pima populations
- The form shows "Pregnancies (0 if male)" — this is misleading.
  The model was NOT trained on male data and results for males
  should not be trusted
- Dataset is from 1988 — medical standards may have changed

## Performance
| Metric | Score |
|---|---|
| Accuracy | 75% |
| ROC-AUC | 0.811 |
| Precision (Diabetes) | 0.66 |
| Recall (Diabetes) | 0.57 |
| F1 Score (Diabetes) | 0.61 |

*Note: Recall is the most clinically important metric —
missing a diabetic (false negative) is worse than a false alarm.*

## Pipeline

## Features
| Feature | Description | Normal Range |
|---|---|---|
| Pregnancies | Number of pregnancies | 0-17 |
| Glucose | Plasma glucose (mg/dL) | 70-140 |
| BloodPressure | Diastolic BP (mm Hg) | 60-90 |
| SkinThickness | Triceps fold (mm) | 10-50 |
| Insulin | 2-hr serum insulin (μU/mL) | 16-166 |
| BMI | Body mass index (kg/m²) | 18.5-40 |
| DiabetesPedigreeFunction | Family history score | 0.08-2.42 |
| Age | Age in years | 21+ |

## Ethical Considerations
- This model should NEVER be used to make real medical decisions
- The Pima dataset has known representation bias
- False negatives (missed diabetics) are more dangerous than
  false positives in this context — model is not optimised for
  clinical use

## Disclaimer
This model is built for **educational purposes only**.
Always consult a qualified medical professional for health advice.