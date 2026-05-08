# ============================================================
# train.py — CLEAN VERSION with proper functions
# Developer: Yashika Sharma | github.com/Yashika41
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split,
    StratifiedKFold, cross_val_score)
from sklearn.metrics import (classification_report,
    roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, accuracy_score)

# Seed for reproducibility
np.random.seed(42)

# Output directory — always save next to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Feature columns that have invalid 0s ──────────────────
ZERO_COLS = ['Glucose', 'BloodPressure',
             'SkinThickness', 'Insulin', 'BMI']

FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure',
                 'SkinThickness', 'Insulin', 'BMI',
                 'DiabetesPedigreeFunction', 'Age']


# ============================================================
# FUNCTION 1 — Load Data
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset and return as DataFrame."""
    df = pd.read_csv(path, header=None, names=FEATURE_NAMES + ['Outcome'])
    print(f"✅ Data loaded! Shape: {df.shape}")
    print(f"   Target distribution:\n{df['Outcome'].value_counts()}")
    return df


# ============================================================
# FUNCTION 2 — Clean Data
# ============================================================
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid 0s with NaN for imputation later."""
    df = df.copy()
    df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)
    missing = df[ZERO_COLS].isnull().sum()
    print(f"\n✅ Zeros replaced with NaN!")
    print(f"   Missing values per column:\n{missing}")
    return df


# ============================================================
# FUNCTION 3 — Split Data
# ============================================================
def split(df: pd.DataFrame):
    """Split into features and target, then train/test sets."""
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\n✅ Split done!")
    print(f"   Training samples : {X_train.shape[0]}")
    print(f"   Testing samples  : {X_test.shape[0]}")
    return X, y, X_train, X_test, y_train, y_test


# ============================================================
# FUNCTION 4 — Train Model
# ============================================================
def train(X_train, y_train) -> Pipeline:
    """Build and train the sklearn Pipeline."""
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        class_weight='balanced'
                    ))
    ])

    pipeline.fit(X_train, y_train)
    print("\n✅ Pipeline trained!")
    print("   Steps:")
    for name, step in pipeline.named_steps.items():
        print(f"   → {name}: {type(step).__name__}")
    return pipeline


# ============================================================
# FUNCTION 5 — Evaluate Model
# ============================================================
def evaluate(pipeline: Pipeline, X_test, y_test,
             X, y) -> dict:
    """Evaluate model with proper medical metrics."""

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc                          = accuracy_score(y_test, y_pred)
    precision, recall, f1, _     = precision_recall_fscore_support(
                                       y_test, y_pred, average='binary')
    auc                          = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 50)
    print("📊 MODEL PERFORMANCE")
    print("=" * 50)
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}  ← most important clinically!")
    print(f"F1 Score  : {f1:.3f}")
    print(f"ROC-AUC   : {auc:.3f}")
    print("=" * 50)
    print("\n📋 Full Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['No Diabetes', 'Diabetes']))

    # Cross Validation
    print("\n📊 Running 5-Fold Cross Validation...")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1  = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
    cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')

    print("=" * 50)
    print("📊 CROSS-VALIDATED SCORES (5 folds)")
    print("=" * 50)
    print(f"F1 Score : {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
    print(f"ROC-AUC  : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
    print("=" * 50)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
    plt.close()
    print("\n✅ Confusion matrix saved!")

    return {
        'accuracy'  : acc,
        'precision' : precision,
        'recall'    : recall,
        'f1'        : f1,
        'roc_auc'   : auc,
        'cv_f1'     : f"{cv_f1.mean():.3f} ± {cv_f1.std():.3f}",
        'cv_auc'    : f"{cv_auc.mean():.3f} ± {cv_auc.std():.3f}"
    }

# ============================================================
# FUNCTION 6 — SHAP Feature Importance Plot
# ============================================================
def shap_plot(pipeline: Pipeline, X_train) -> None:
    """Generate SHAP feature importance plot."""
    print("\n📊 Generating SHAP feature importance plot...")

    # Get the trained model from pipeline
    rf_model = pipeline.named_steps['model']

    # Transform training data through imputer + scaler only
    X_train_transformed = pipeline[:-1].transform(X_train)

    # Create SHAP explainer
    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_train_transformed)

    # Plot — use index 1 for Diabetes class
    plt.figure()
    shap.summary_plot(
        shap_values[:, :, 1] if len(
            np.array(shap_values).shape) == 3 else shap_values[1],
        X_train_transformed,
        feature_names=FEATURE_NAMES,
        show=False
    )
    plt.title('Feature Importance — SHAP Values')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'shap_importance.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("✅ SHAP plot saved as shap_importance.png!")

# ============================================================
# FUNCTION 7 — Save Artifacts
# ============================================================
def save_artifacts(pipeline: Pipeline) -> None:
    """Save trained pipeline to disk."""
    path = os.path.join(BASE_DIR, 'pipeline.pkl')
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\n✅ Pipeline saved to: {path}")


# ============================================================
# MAIN — Run everything
# ============================================================
if __name__ == '__main__':
    print("🚀 Starting Disease Prediction ML Pipeline...\n")

    # Step 1 — Load
    df = load_data(os.path.join(BASE_DIR, 'diabetes.csv'))

    # Step 2 — Clean
    df = clean(df)

    # Step 3 — Split
    X, y, X_train, X_test, y_train, y_test = split(df)

    # Step 4 — Train
    pipeline = train(X_train, y_train)

    # Step 5 — Evaluate
    metrics = evaluate(pipeline, X_test, y_test, X, y)

    # Step 6 — SHAP plot
    shap_plot(pipeline, X_train)

    # Step 7 — Save
    save_artifacts(pipeline)

    print("\n" + "=" * 50)
    print("🎉 Pipeline complete! Now run: python app.py")
    print("=" * 50)