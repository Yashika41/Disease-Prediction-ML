# train.py — Run this on your laptop to create pipeline.pkl
import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import os

# ── Load data ──────────────────────────────────────────
df = pd.read_csv('diabetes.csv', header=None,
     names=['Pregnancies','Glucose','BloodPressure',
            'SkinThickness','Insulin','BMI',
            'DiabetesPedigreeFunction','Age','Outcome'])
print(f"✅ Data loaded! Shape: {df.shape}")

# ── Replace 0s with NaN ────────────────────────────────
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols] = df[cols].replace(0, np.nan)
print("✅ Zeros replaced with NaN!")

# ── Split ──────────────────────────────────────────────
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Split done! Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ── Build & Train Pipeline ─────────────────────────────
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced'))
])
pipeline.fit(X_train, y_train)
print("✅ Pipeline trained!")

# ── Evaluate ───────────────────────────────────────────
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]
auc     = roc_auc_score(y_test, y_proba)
print("\n📊 Results:")
print(classification_report(y_test, y_pred,
      target_names=['No Diabetes','Diabetes']))
print(f"ROC-AUC: {auc:.3f}")

# ── Save ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'pipeline.pkl'), 'wb') as f:
    pickle.dump(pipeline, f)
print("\n✅ pipeline.pkl saved successfully!")
# ── Cross Validation ───────────────────────────────────
print("\n📊 Running 5-Fold Cross Validation...")
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1  = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')

print("=" * 45)
print("📊 CROSS-VALIDATED SCORES (5 folds)")
print("=" * 45)
print(f"F1 Score : {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
print(f"ROC-AUC  : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
print("=" * 45)
print("✅ These are your REAL scores!")
print("🎉 Now run: python app.py")