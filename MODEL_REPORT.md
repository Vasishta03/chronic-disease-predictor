# Chronic Disease Prediction Models - Performance Report

## Model Training Summary

### Diabetes Model
- **Algorithm**: RandomForestClassifier
- **Accuracy**: 0.740
- **Precision**: 0.735
- **Recall**: 0.446
- **F1-Score**: 0.556
- **ROC-AUC**: 0.835

### Heart Disease Model
- **Algorithm**: GradientBoostingClassifier
- **Accuracy**: 0.721
- **Precision**: 0.719
- **Recall**: 0.742
- **F1-Score**: 0.730
- **ROC-AUC**: 0.799

### Hypertension Model
- **Algorithm**: LogisticRegression
- **Accuracy**: 0.740
- **Precision**: 0.833
- **Recall**: 0.294
- **F1-Score**: 0.435
- **ROC-AUC**: 0.858

## Feature Importance

### Diabetes - Top 5 Features
1. **glucose**: 0.331
2. **bmi**: 0.140
3. **diabetes_pedigree**: 0.125
4. **age**: 0.106
5. **blood_pressure**: 0.092

### Heart Disease - Top 5 Features
1. **cholesterol**: 0.234
2. **exercise_angina**: 0.200
3. **age**: 0.168
4. **resting_bp**: 0.119
5. **chest_pain_type**: 0.088

## Model Files

The following files have been generated:
- `models/diabetes_model.pkl` - Diabetes prediction model
- `models/heart_disease_model.pkl` - Heart disease prediction model
- `models/hypertension_model.pkl` - Hypertension prediction model
- `models/hypertension_scaler.pkl` - Feature scaler for hypertension model
- `models/feature_names.pkl` - Feature names for all models
- `models/performance_metrics.pkl` - Performance metrics for all models
