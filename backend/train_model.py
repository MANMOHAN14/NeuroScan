"""
Model Training Pipeline for Learning Disabilities Detection
=============================================================
Trains and evaluates multiple ML models, selects the best one,
and saves it along with preprocessing artifacts.

Models Compared:
    - Random Forest Classifier
    - XGBoost Classifier
    - Gradient Boosting Classifier

Evaluation Metrics:
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix
    - Feature Importance
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("  [WARNING] XGBoost not installed. Skipping XGBoost model.")


# ─── Configuration ───────────────────────────────────────────
DATA_PATH = os.path.join('data', 'learning_disabilities.csv')
MODEL_DIR = 'models'
PLOTS_DIR = os.path.join('models', 'plots')

DISABILITY_LABELS = {
    0: "No Disability",
    1: "Dyslexia",
    2: "Dysgraphia",
    3: "Dyscalculia",
    4: "ADHD"
}

TARGET_COL = 'disability_type'
CATEGORICAL_COLS = ['gender', 'parent_education', 'family_history_ld']
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_preprocess():
    """Load dataset and perform preprocessing."""
    print("\n  [1/5] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"        Shape: {df.shape}")
    print(f"        Columns: {list(df.columns)}")

    # Separate features and target
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # Encode categorical features
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                print(f"        Encoded '{col}': {list(le.classes_)}")
            else:
                print(f"        '{col}' already numeric (dtype={X[col].dtype})")

    # Ensure all columns are numeric before scaling
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"        [WARNING] Converting non-numeric column '{col}' to numeric")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X.columns.tolist()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    print(f"        Scaled {len(numerical_cols)} numerical features")

    return X, y, scaler, label_encoders, X.columns.tolist()


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and compare performance."""
    print("\n  [2/5] Training models...")

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        ),
    }

    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
            use_label_encoder=False
        )

    results = {}

    for name, model in models.items():
        print(f"\n        Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'report': classification_report(y_test, y_pred,
                                           target_names=list(DISABILITY_LABELS.values()))
        }

        print(f"        [OK] Accuracy: {accuracy:.4f} | F1: {f1:.4f} | "
              f"CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    return results


def select_best_model(results):
    """Select the best model based on F1 score."""
    print("\n  [3/5] Selecting best model...")

    best_name = max(results, key=lambda k: results[k]['f1_score'])
    best = results[best_name]

    print(f"        [BEST] Best Model: {best_name}")
    print(f"          Accuracy: {best['accuracy']:.4f}")
    print(f"          F1 Score: {best['f1_score']:.4f}")
    print(f"          CV Score: {best['cv_mean']:.4f} +/- {best['cv_std']:.4f}")
    print(f"\n        Classification Report:\n{best['report']}")

    return best_name, best


def generate_plots(results, best_name, y_test, feature_names):
    """Generate evaluation plots."""
    print("\n  [4/5] Generating evaluation plots...")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    best = results[best_name]
    model = best['model']

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, best['y_pred'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(DISABILITY_LABELS.values()),
                yticklabels=list(DISABILITY_LABELS.values()))
    plt.title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("        [OK] Saved confusion_matrix.png")

    # --- Model Comparison ---
    model_names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in model_names]
    f1_scores = [results[n]['f1_score'] for n in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy',
                   color='#6366f1', alpha=0.85)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score',
                   color='#06b6d4', alpha=0.85)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    print("        [OK] Saved model_comparison.png")

    # --- Feature Importance ---
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
        plt.barh(range(len(feature_names)),
                 importances[indices][::-1],
                 color=colors)
        plt.yticks(range(len(feature_names)),
                   [feature_names[i] for i in indices][::-1])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Feature Importance - {best_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150)
        plt.close()
        print("        [OK] Saved feature_importance.png")


def save_model(best_name, best_result, scaler, label_encoders, feature_names):
    """Save the best model and preprocessing artifacts."""
    print("\n  [5/5] Saving model and artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    model = best_result['model']

    # Save model
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"        [OK] Model saved: {model_path}")

    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"        [OK] Scaler saved: {scaler_path}")

    # Save label encoders
    encoders_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
    joblib.dump(label_encoders, encoders_path)
    print(f"        [OK] Encoders saved: {encoders_path}")

    # Save feature importance
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_names,
                                      model.feature_importances_.tolist()))

    # Save metadata
    metadata = {
        'model_name': best_name,
        'accuracy': round(best_result['accuracy'], 4),
        'f1_score': round(best_result['f1_score'], 4),
        'cv_mean': round(best_result['cv_mean'], 4),
        'cv_std': round(best_result['cv_std'], 4),
        'features': feature_names,
        'feature_importance': feature_importance,
        'classes': DISABILITY_LABELS,
        'n_features': len(feature_names),
        'n_classes': len(DISABILITY_LABELS),
    }

    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"        [OK] Metadata saved: {metadata_path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  Learning Disabilities Detection - Model Training")
    print("=" * 60)

    # Load & preprocess
    X, y, scaler, label_encoders, feature_names = load_and_preprocess()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n        Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Train models
    results = train_models(X_train, y_train, X_test, y_test)

    # Select best
    best_name, best_result = select_best_model(results)

    # Generate plots
    generate_plots(results, best_name, y_test, feature_names)

    # Save model
    save_model(best_name, best_result, scaler, label_encoders, feature_names)

    print("\n" + "=" * 60)
    print(f"  [OK] Training complete! Best model: {best_name}")
    print(f"    Accuracy: {best_result['accuracy']:.4f}")
    print(f"    F1 Score: {best_result['f1_score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
