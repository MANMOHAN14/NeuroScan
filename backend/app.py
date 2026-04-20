"""
Flask REST API for Learning Disabilities Detection
====================================================
Serves the trained ML model via HTTP endpoints for
the frontend to consume.

Endpoints:
    POST /api/predict        - Get prediction for student data
    GET  /api/model-info     - Get model metadata
    GET  /api/feature-importance - Get feature importances
    GET  /api/health         - Health check
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─── App Setup ───────────────────────────────────────────
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# ─── Load Model Artifacts ────────────────────────────────
MODEL_DIR = 'models'

print("  Loading model artifacts...")
model = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))

with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
    metadata = json.load(f)

print(f"  [OK] Loaded {metadata['model_name']} (Accuracy: {metadata['accuracy']})")

DISABILITY_LABELS = {
    0: "No Disability",
    1: "Dyslexia",
    2: "Dysgraphia",
    3: "Dyscalculia",
    4: "ADHD"
}

DISABILITY_DESCRIPTIONS = {
    "No Disability": {
        "description": "No significant learning disability indicators detected.",
        "recommendations": [
            "Continue providing a supportive learning environment",
            "Maintain regular academic monitoring",
            "Encourage balanced study habits and extracurricular activities"
        ],
        "color": "#10b981"
    },
    "Dyslexia": {
        "description": "Dyslexia is a learning disorder that involves difficulty reading due to problems identifying speech sounds and learning how they relate to letters and words.",
        "recommendations": [
            "Use multisensory teaching methods (visual, auditory, kinesthetic)",
            "Provide audiobooks and text-to-speech tools",
            "Allow extra time for reading assignments and tests",
            "Use structured literacy programs (Orton-Gillingham approach)",
            "Consult with a reading specialist for formal assessment"
        ],
        "color": "#f59e0b"
    },
    "Dysgraphia": {
        "description": "Dysgraphia is a neurological disorder characterized by writing difficulties, including impaired handwriting, spelling, and composition.",
        "recommendations": [
            "Allow use of keyboards/computers for written assignments",
            "Provide occupational therapy for fine motor skill development",
            "Use graphic organizers for writing tasks",
            "Reduce volume of written work while maintaining learning objectives",
            "Practice hand-strengthening exercises"
        ],
        "color": "#8b5cf6"
    },
    "Dyscalculia": {
        "description": "Dyscalculia is a learning disability that affects math ability, including understanding numbers, learning math facts, and performing calculations.",
        "recommendations": [
            "Use visual aids and manipulatives for math concepts",
            "Break problems into smaller, manageable steps",
            "Allow use of calculators and math reference sheets",
            "Provide extra practice with real-world math applications",
            "Consider specialized math tutoring programs"
        ],
        "color": "#ec4899"
    },
    "ADHD": {
        "description": "ADHD (Attention-Deficit/Hyperactivity Disorder) is a neurodevelopmental disorder characterized by persistent patterns of inattention, hyperactivity, and impulsivity.",
        "recommendations": [
            "Create a structured and predictable classroom routine",
            "Provide frequent breaks during long tasks",
            "Use positive reinforcement and behavior management strategies",
            "Minimize distractions in the learning environment",
            "Consult with a healthcare professional for comprehensive evaluation"
        ],
        "color": "#ef4444"
    }
}

FEATURE_ORDER = metadata['features']

# ─── Routes ──────────────────────────────────────────────


@app.route('/')
def serve_frontend():
    """Serve the frontend application."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": metadata['model_name']
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model metadata and performance metrics."""
    return jsonify({
        "model_name": metadata['model_name'],
        "accuracy": metadata['accuracy'],
        "f1_score": metadata['f1_score'],
        "cv_mean": metadata['cv_mean'],
        "cv_std": metadata['cv_std'],
        "n_features": metadata['n_features'],
        "n_classes": metadata['n_classes'],
        "features": metadata['features'],
        "classes": DISABILITY_LABELS
    })


@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    """Return feature importance values."""
    importance = metadata.get('feature_importance', {})
    # Sort by importance (descending)
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )
    return jsonify(sorted_importance)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction based on student data."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        missing = [f for f in FEATURE_ORDER if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing features: {missing}",
                "required_features": FEATURE_ORDER
            }), 400

        # Build input DataFrame in correct feature order
        input_data = {}
        for feature in FEATURE_ORDER:
            input_data[feature] = [data[feature]]

        input_df = pd.DataFrame(input_data)

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col])
                except ValueError:
                    # Handle unseen labels
                    input_df[col] = 0

        # Scale features
        numerical_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Predict
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        predicted_label = DISABILITY_LABELS[int(prediction)]
        confidence = float(probabilities[int(prediction)]) * 100

        # Determine risk level
        if predicted_label == "No Disability":
            risk_level = "Low"
        elif confidence >= 70:
            risk_level = "High"
        elif confidence >= 50:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        # Build response
        class_probabilities = {}
        for i, label in DISABILITY_LABELS.items():
            class_probabilities[label] = round(float(probabilities[i]) * 100, 2)

        disability_info = DISABILITY_DESCRIPTIONS.get(predicted_label, {})

        response = {
            "prediction": predicted_label,
            "confidence": round(confidence, 2),
            "risk_level": risk_level,
            "class_probabilities": class_probabilities,
            "description": disability_info.get("description", ""),
            "recommendations": disability_info.get("recommendations", []),
            "color": disability_info.get("color", "#6366f1"),
            "input_summary": {
                "age": data.get('age'),
                "reading_score": data.get('reading_score'),
                "writing_score": data.get('writing_score'),
                "math_score": data.get('math_score'),
                "attention_span": data.get('attention_span'),
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Main ────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Learning Disabilities Detection API")
    print(f"  Model: {metadata['model_name']}")
    print(f"  Accuracy: {metadata['accuracy']}")
    print("  Server: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
