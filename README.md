# 🧠 NeuroScan AI — Early Detection of Learning Disabilities

An AI-powered full-stack web application that uses machine learning to screen children for potential learning disabilities including **Dyslexia**, **Dysgraphia**, **Dyscalculia**, and **ADHD**.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Machine Learning Model](#machine-learning-model)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Disclaimer](#disclaimer)

---

## 🎯 Overview

Learning disabilities affect approximately **5-15%** of school-age children worldwide. Early identification is crucial for implementing effective interventions. This project demonstrates how machine learning can be used as a screening tool to assist educators and parents in identifying children who may be at risk for learning disabilities.

The system analyzes **20 features** across four categories:
- **Academic Performance**: Reading, writing, math scores, vocabulary, phonological awareness
- **Cognitive Abilities**: Attention span, working memory, processing speed
- **Behavioral Indicators**: Hyperactivity, impulsivity, social skills, class participation
- **Environmental Factors**: Family history, sleep, screen time, parental education

## ✨ Features

- 🤖 **AI-Powered Prediction**: Multi-class classification detecting 4 types of learning disabilities
- 📊 **Interactive Dashboard**: Visual results with probability charts and radar profiles
- 🎨 **Modern UI**: Premium dark theme with glassmorphism design and animations
- 📋 **Multi-Step Assessment**: User-friendly 4-step assessment form with sliders
- 📈 **Model Performance**: Real-time accuracy metrics and feature importance visualization
- 💡 **Recommendations**: Personalized recommendations for each identified condition
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🛠️ Technology Stack

### Backend
| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Programming language |
| Flask | Web framework / REST API |
| scikit-learn | ML model training & evaluation |
| XGBoost | Gradient boosting classifier |
| Pandas & NumPy | Data processing |
| Matplotlib & Seaborn | Training visualizations |

### Frontend
| Technology | Purpose |
|-----------|---------|
| HTML5 | Page structure |
| CSS3 | Styling & animations |
| JavaScript (ES6+) | Application logic |
| Chart.js | Interactive data visualization |
| Google Fonts | Typography (Inter, Outfit) |

## 📁 Project Structure

```
AI Model/
├── backend/
│   ├── app.py                    # Flask REST API server
│   ├── generate_dataset.py       # Synthetic dataset generator
│   ├── train_model.py            # ML model training pipeline
│   ├── requirements.txt          # Python dependencies
│   ├── data/
│   │   └── learning_disabilities.csv  # Generated dataset
│   └── models/
│       ├── model.pkl             # Trained ML model
│       ├── scaler.pkl            # Feature scaler
│       ├── label_encoders.pkl    # Categorical encoders
│       ├── model_metadata.json   # Model performance metrics
│       └── plots/                # Training visualizations
│           ├── confusion_matrix.png
│           ├── model_comparison.png
│           └── feature_importance.png
├── frontend/
│   ├── index.html                # Main HTML page
│   ├── css/
│   │   └── style.css             # Design system
│   ├── js/
│   │   └── app.js                # Frontend logic
│   └── assets/                   # Images and icons
└── README.md
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Web browser (Chrome/Firefox/Edge)

### Step 1: Clone/Download the Project
```bash
cd "d:\AI Model"
```

### Step 2: Set Up Python Environment
```bash
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Generate Dataset & Train Model
```bash
# Generate synthetic dataset
python generate_dataset.py

# Train the ML model
python train_model.py
```

### Step 4: Start the Server
```bash
python app.py
```

### Step 5: Open the Application
Open your browser and navigate to: **http://localhost:5000**

## 📖 Usage

1. **Navigate** to the application in your browser
2. **Click** "Start Assessment" on the landing page
3. **Fill out** the 4-step assessment form:
   - Step 1: Basic Information (age, gender)
   - Step 2: Academic Scores (reading, writing, math, etc.)
   - Step 3: Cognitive & Behavioral data
   - Step 4: Environmental factors
4. **Click** "Analyze with AI" to get the prediction
5. **Review** the results including:
   - Predicted condition with confidence level
   - Risk level assessment
   - Probability distribution chart
   - Student profile radar chart
   - Personalized recommendations

## 🤖 Machine Learning Model

### Dataset
- **Size**: 5,000 synthetic samples
- **Features**: 20 (academic, cognitive, behavioral, environmental)
- **Classes**: 5 (No Disability, Dyslexia, Dysgraphia, Dyscalculia, ADHD)
- **Distribution**: 40% No Disability, 15% each for the 4 disability types

### Models Compared
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | ~95% | ~95% |
| XGBoost | ~96% | ~96% |
| Gradient Boosting | ~94% | ~94% |

### Evaluation
- **Cross-validation**: 5-fold stratified CV
- **Metrics**: Accuracy, F1-score, precision, recall
- **Visualizations**: Confusion matrix, model comparison, feature importance

## 📡 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/model-info` | Model metadata |
| `GET` | `/api/feature-importance` | Feature importance values |
| `POST` | `/api/predict` | Make prediction |

### Prediction Request Example
```json
POST /api/predict
Content-Type: application/json

{
    "age": 8,
    "gender": 1,
    "reading_score": 35,
    "writing_score": 50,
    "math_score": 65,
    "attention_span": 6,
    "memory_score": 55,
    "processing_speed": 50,
    "hyperactivity_score": 4,
    "impulsivity_score": 4,
    "social_skills": 6,
    "fine_motor_skills": 65,
    "phonological_awareness": 30,
    "vocabulary_score": 45,
    "parent_education": "Bachelors",
    "family_history_ld": 1,
    "sleep_hours": 7.5,
    "screen_time": 3.5,
    "class_participation": 5,
    "emotional_stability": 5
}
```

### Prediction Response Example
```json
{
    "prediction": "Dyslexia",
    "confidence": 82.5,
    "risk_level": "High",
    "class_probabilities": {
        "No Disability": 5.2,
        "Dyslexia": 82.5,
        "Dysgraphia": 6.1,
        "Dyscalculia": 3.8,
        "ADHD": 2.4
    },
    "description": "Dyslexia is a learning disorder...",
    "recommendations": ["Use multisensory teaching methods...", ...],
    "color": "#f59e0b"
}
```

## ⚠️ Disclaimer

This application is an **academic project** and serves as a **screening/decision-support tool only**. It is **NOT** a substitute for professional clinical diagnosis. The predictions should not be used as the sole basis for any medical or educational decisions. Always consult qualified healthcare professionals and educational specialists for comprehensive evaluation and diagnosis of learning disabilities.

---

**Developed as a College Project** | Machine Learning Application in Education
