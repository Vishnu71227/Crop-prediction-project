# 🌾 Crop Recommendation System

An end-to-end Machine Learning web application that recommends the best crop to grow based on soil nutrients and climate conditions.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![Accuracy](https://img.shields.io/badge/Accuracy-99.3%25-brightgreen)

---

## 📌 Project Overview

Farmers often struggle to decide which crop to grow based on their soil and climate conditions. This project solves that problem using Machine Learning — a farmer simply enters soil nutrient values and climate data, and the system recommends the most suitable crop.

- **Dataset:** 2200 samples | 22 crop types | 7 input features
- **Best Model:** Random Forest Classifier
- **Accuracy:** 99.3% (Test) | 99.45% ± 0.34% (5-Fold Cross Validation)
- **Deployment:** Streamlit Web Application

---

## 🎯 Problem Statement

> Given soil nutrient levels (N, P, K), temperature, humidity, pH, and rainfall — predict the most suitable crop to grow.

---

## 📊 Dataset Information

| Feature | Description | Unit |
|---|---|---|
| N | Nitrogen content in soil | mg/kg |
| P | Phosphorus content in soil | mg/kg |
| K | Potassium content in soil | mg/kg |
| temperature | Average temperature | °C |
| humidity | Relative humidity | % |
| ph | pH value of soil | 0–14 |
| rainfall | Annual rainfall | mm |
| label | Crop name (target) | — |

- **Total Samples:** 2200
- **Total Crops:** 22 (rice, maize, chickpea, mango, apple, cotton, coffee, etc.)
- **Class Balance:** 100 samples per crop (perfectly balanced ✅)
- **Missing Values:** None ✅

---

## 🤖 Models Used & Comparison

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Random Forest | 100.00% | **99.32%** ✅ Chosen |
| Decision Tree | 100.00% | 98.64% |
| Naive Bayes | 99.49% | 99.55% |

**Why Random Forest?**
- Ensemble method — 100 trees vote for final prediction
- Overfitting resistant compared to single Decision Tree
- Provides Feature Importance for business insights
- Industry standard for tabular classification tasks

---

## 🔍 Key Insights

### Feature Importance (What matters most?)

| Rank | Feature | Importance |
|---|---|---|
| 🥇 1 | Rainfall | 22.70% |
| 🥈 2 | Humidity | 21.13% |
| 🥉 3 | Potassium (K) | 18.12% |
| 4 | Phosphorus (P) | 14.36% |
| 5 | Nitrogen (N) | 10.89% |
| 6 | Temperature | 7.57% |
| 7 | pH | 5.23% |

> Rainfall and Humidity are the most influential factors in crop selection.

### Cross Validation Results
```
5-Fold CV Accuracy: 99.45% ± 0.34%
→ Low standard deviation confirms model is stable and not overfitting ✅
```

---

## 🧠 SHAP Explainability

This project includes **SHAP (SHapley Additive exPlanations)** to make the model transparent and interpretable.

- SHAP explains *why* the model made a specific prediction
- Shows each feature's contribution (positive or negative) to the prediction
- Makes the model trustworthy for real-world agricultural use

> *"Rainfall and Humidity contributed most positively to crop predictions — directly aligning with agricultural domain knowledge."*

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (RandomForest, DecisionTree, NaiveBayes) |
| Explainability | SHAP |
| Model Saving | Joblib |
| Web Framework | Streamlit |

---

## 📁 Project Structure

```
Crop-prediction-project/
│
├── task4_farmer.ipynb              ← Main ML notebook (EDA + Modeling + SHAP)
├── streamlit_app.py                ← Streamlit web application
├── farmer.csv                      ← Raw dataset
├── crop_recommendation_model.pkl   ← Trained Random Forest model
├── crop_label_encoder.pkl          ← Label Encoder (numbers → crop names)
├── requirements.txt                ← All dependencies
└── README.md                       ← Project documentation
```

---

## ⚙️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Vishnu1234-cloud/Crop-prediction-project.git
cd Crop-prediction-project
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

### 5. Open in Browser
```
http://localhost:8501
```

---

## 🌐 App Features

| Feature | Description |
|---|---|
| 🎚️ Interactive Sliders | 7 input features with sliders |
| 📊 Input Summary Table | Shows feature values + importance % |
| 🌱 Crop Prediction | Instant crop recommendation |
| 💡 Crop Tips | Farming tip for each recommended crop |
| 📈 Probability Chart | Top 5 crop probabilities with bar chart |
| ℹ️ Sidebar | Model info + all 22 supported crops |

---

## 🌱 Sample Prediction

| Input | Value |
|---|---|
| Nitrogen (N) | 90 |
| Phosphorus (P) | 42 |
| Potassium (K) | 43 |
| Temperature | 21.0°C |
| Humidity | 82% |
| pH | 6.5 |
| Rainfall | 200mm |

**Output:** 🌾 Recommended Crop: **RICE**
> 🌊 Rice needs high water — ensure proper irrigation.

---

## 📈 ML Pipeline

```
Raw CSV Data
    ↓
EDA (Null check, Duplicates, Distribution, Correlation Heatmap, Boxplots)
    ↓
Label Encoding (crop names → numbers)
    ↓
Train/Test Split (80% / 20%)
    ↓
Model Training (Random Forest, Decision Tree, Naive Bayes)
    ↓
Model Comparison + Best Model Selection
    ↓
Evaluation (Accuracy, Classification Report, Confusion Matrix)
    ↓
Cross Validation (5-Fold)
    ↓
Feature Importance + SHAP Explainability
    ↓
Model Saved (.pkl)
    ↓
Streamlit Web App Deployment
```

---

## 👨‍💻 Author

**Vishnu Sharma**
- 🎓 B.Tech Information Technology — JECRC Foundation, Jaipur (2023–2027)
- 💼 Data Science & ML Intern — Upflairs
- 🔗 GitHub: [Vishnu1234-cloud](https://github.com/Vishnu1234-cloud)
- 📧 LinkedIn: [Add your LinkedIn URL]

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
