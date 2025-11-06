# Chronic Disease Prediction using Classical Machine Learning
An AI-powered health screening system that predicts risk for three major chronic diseases (diabetes, heart disease, 
and hypertension) using interpretable classical machine learning algorithms. The system provides real-time risk assessment 
with explainable predictions through a user-friendly Streamlit dashboard.

---

## Overview

This project is a production-ready, interpretable machine learning system for predicting the risk of three major chronic diseases:
- **Diabetes**
- **Heart Disease**
- **Hypertension**

It uses **classical machine learning algorithms** (Random Forest, Gradient Boosting, Logistic Regression) and features a user-friendly, privacy-focused **Streamlit dashboard** for real-time, explainable predictions.

---

## Key Features

- **Multi-disease Prediction:** Risk profiling for diabetes, heart disease, and hypertension.
- **Explainable AI:** Shows which factors (e.g., glucose, cholesterol, family history) drive each prediction.
- **Offline Capable:** Works fully offline once setup is complete.
- **Privacy-First:** All processing is local. No patient data is sent externally (HIPAA/GDPR aligned).
- **Automated Setup:** One-command pipeline for model training and app launch.
- **Ready for Clinics and Research:** Suitable for point-of-care screening, rural health, and researcher use.
- **Mobile-Responsive Dashboard:** Streamlit web app works on desktop and mobile.

---

## Datasets Used

- **Diabetes:** [PIMA Indian Diabetes Database (UCI)](https://archive.ics.uci.edu/ml/datasets/Diabetes)
- **Heart Disease:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Hypertension:** Synthetic dataset structured to reflect CDC BRFSS standards

---

## System Architecture

1. **Data Collection & Ingestion**
2. **Preprocessing & Feature Engineering:** Imputation, scaling, encoding, BMI calculation
3. **Model Training & Validation:** Grid search with cross-validation
4. **Model Serialization & Storage:** Pickle/save best models locally
5. **Streamlit Dashboard Interface:** Interactive web UI for input/output
6. **Real-time Prediction & Explanation:** Risk score + feature importance visualization

---

## Performance Summary

| Disease       | Model              | Accuracy | F1-Score | ROC-AUC |
|---------------|--------------------|----------|----------|---------|
| Diabetes      | Random Forest      | 0.73     | 0.55     | 0.82    |
| Heart Disease | Gradient Boosting  | 0.72     | 0.73     | 0.81    |
| Hypertension  | Logistic Regression| 0.82     | 0.73     | 0.86    |

---

## Setup Instructions

1. **Clone the repo**
    ```
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

3. **Train the models**
    ```
    python train_models.py
    ```

4. **Run the Streamlit dashboard**
    ```
    streamlit run app.py
    ```

5. **Open your browser**
    - Navigate to [http://localhost:8501](http://localhost:8501)

---


## Future Enhancements

- Integration with electronic health record (EHR) systems
- SHAP value-based advanced interpretability
- Support for additional conditions (e.g., kidney disease)
- Mobile app deployment
- Clinical validation studies with healthcare partners

---

## Citation

If you use this project in academic work, please cite: Vasishta Nandipati
EMAIL: vasishtavj@gmail.com
