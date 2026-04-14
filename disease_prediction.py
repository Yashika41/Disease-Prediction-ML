# ============================================================
# DISEASE PREDICTION ML PROJECT - STARTER CODE
# Yashika Sharma | github.com/Yashika41
# ============================================================

# -----------------------------------------------
# STEP 1: INSTALL LIBRARIES (run in terminal first)
# -----------------------------------------------
# pip install pandas numpy scikit-learn matplotlib seaborn flask

# -----------------------------------------------
# STEP 2: IMPORT LIBRARIES
# -----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")


# -----------------------------------------------
# STEP 3: LOAD THE DATASET
# -----------------------------------------------
# Download from Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# Save it as 'diabetes.csv' in the same folder as this file

df = pd.read_csv('diabetes.csv')

print("\n📊 Dataset Shape:", df.shape)
print("\n📋 First 5 rows:")
print(df.head())


# -----------------------------------------------
# STEP 4: EXPLORE THE DATA
# -----------------------------------------------
print("\n📈 Basic Statistics:")
print(df.describe())

print("\n❓ Missing Values:")
print(df.isnull().sum())

print("\n🎯 Target column (Outcome) value counts:")
print(df['Outcome'].value_counts())
# 0 = No Diabetes, 1 = Diabetes


# -----------------------------------------------
# STEP 5: VISUALIZE THE DATA
# -----------------------------------------------

# Plot 1: Count of diabetic vs non-diabetic
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df, palette='Set2')
plt.title('Diabetes Distribution (0 = No, 1 = Yes)')
plt.savefig('diabetes_distribution.png')
plt.show()
print("✅ Chart saved!")

# Plot 2: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("✅ Heatmap saved!")


# -----------------------------------------------
# STEP 6: HANDLE ZERO VALUES (DATA CLEANING)
# -----------------------------------------------
# Some columns have 0 as placeholder for missing values
# (you can't have 0 glucose or 0 blood pressure)

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].median())

print("✅ Zero values replaced with median values!")
print(df.describe())


# -----------------------------------------------
# STEP 7: SPLIT DATA INTO FEATURES AND TARGET
# -----------------------------------------------
X = df.drop('Outcome', axis=1)   # All columns except the result
y = df['Outcome']                  # The result column (0 or 1)

print("\n✅ Features (X) shape:", X.shape)
print("✅ Target (y) shape:", y.shape)
print("\n📋 Feature columns:", list(X.columns))


# -----------------------------------------------
# STEP 8: SPLIT INTO TRAINING AND TESTING SETS
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% training, 20% testing
    random_state=42,    # For reproducibility
    stratify=y          # Keeps same ratio in both sets
)

print(f"\n✅ Training set size: {X_train.shape[0]} samples")
print(f"✅ Testing set size:  {X_test.shape[0]} samples")


# -----------------------------------------------
# STEP 9: SCALE THE FEATURES
# -----------------------------------------------
# Scaling makes all features on the same range
# This helps the model learn better

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("\n✅ Features scaled successfully!")


# -----------------------------------------------
# STEP 10: TRAIN MODEL 1 — LOGISTIC REGRESSION
# -----------------------------------------------
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f"\n🔵 Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")


# -----------------------------------------------
# STEP 11: TRAIN MODEL 2 — RANDOM FOREST
# -----------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"🟢 Random Forest Accuracy:       {rf_accuracy * 100:.2f}%")


# -----------------------------------------------
# STEP 12: EVALUATE THE BEST MODEL
# -----------------------------------------------
# Use whichever gave better accuracy
best_model = rf_model if rf_accuracy >= lr_accuracy else lr_model
best_predictions = rf_predictions if rf_accuracy >= lr_accuracy else lr_predictions
best_name = "Random Forest" if rf_accuracy >= lr_accuracy else "Logistic Regression"

print(f"\n🏆 Best Model: {best_name}")
print(f"\n📊 Classification Report:")
print(classification_report(y_test, best_predictions, target_names=['No Diabetes', 'Diabetes']))

# Confusion Matrix
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("✅ Confusion matrix saved!")


# -----------------------------------------------
# STEP 13: TEST WITH A CUSTOM INPUT
# -----------------------------------------------
# Try predicting for a new patient
# Format: [Pregnancies, Glucose, BloodPressure, SkinThickness,
#          Insulin, BMI, DiabetesPedigreeFunction, Age]

sample_patient = np.array([[2, 130, 70, 30, 100, 28.5, 0.45, 25]])
sample_scaled  = scaler.transform(sample_patient)
prediction     = best_model.predict(sample_scaled)
probability    = best_model.predict_proba(sample_scaled)

result = "🔴 Diabetic" if prediction[0] == 1 else "🟢 Not Diabetic"
print(f"\n🧪 Sample Patient Prediction: {result}")
print(f"   Probability — No Diabetes: {probability[0][0]*100:.1f}%")
print(f"   Probability — Diabetes:    {probability[0][1]*100:.1f}%")


# -----------------------------------------------
# STEP 14: SAVE THE TRAINED MODEL
# -----------------------------------------------
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ Model saved as 'model.pkl'")
print("✅ Scaler saved as 'scaler.pkl'")
print("\n🎉 ML part complete! Now move to Flask (app.py) to build the web interface.")
