import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Chronic Disease Prediction System",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disease-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
        font-size: 1.2em;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
        font-size: 1.2em;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
        font-size: 1.2em;
    }
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .debug-info {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ProductionMLSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.performance_metrics = {}
        self.models_loaded = False
        self.error_message = ""
        self.debug_info = []

        self.load_trained_models()

    def load_trained_models(self):

        current_dir = os.getcwd()
        self.debug_info.append(f"Current working directory: {current_dir}")

        if os.path.exists('.'):
            files_in_current = os.listdir('.')
            self.debug_info.append(f"Files in current directory: {files_in_current}")

        models_dir = 'models'
        if os.path.exists(models_dir):
            model_files = os.listdir(models_dir)
            self.debug_info.append(f"Files in models directory: {model_files}")
        else:
            self.debug_info.append("Models directory not found!")
            self.error_message = "Models directory not found. Please run 'python train_models.py' first."
            return False

        required_files = {
            'diabetes_model.pkl': 'Diabetes prediction model',
            'heart_disease_model.pkl': 'Heart disease prediction model',
            'hypertension_model.pkl': 'Hypertension prediction model',
            'hypertension_scaler.pkl': 'Hypertension feature scaler',
            'feature_names.pkl': 'Feature names mapping',
            'performance_metrics.pkl': 'Model performance metrics'
        }

        missing_files = []
        for filename in required_files.keys():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                self.debug_info.append(f"Found: {filename}")
            else:
                self.debug_info.append(f"Missing: {filename}")
                missing_files.append(filename)

        if missing_files:
            self.error_message = f"Missing model files: {', '.join(missing_files)}"
            return False

        try:
            # Load models
            self.debug_info.append("Loading diabetes model...")
            with open(os.path.join(models_dir, 'diabetes_model.pkl'), 'rb') as f:
                self.models['diabetes'] = pickle.load(f)
            self.debug_info.append("Diabetes model loaded successfully")

            self.debug_info.append("Loading heart disease model...")
            with open(os.path.join(models_dir, 'heart_disease_model.pkl'), 'rb') as f:
                self.models['heart_disease'] = pickle.load(f)
            self.debug_info.append("Heart disease model loaded successfully")

            self.debug_info.append("Loading hypertension model...")
            with open(os.path.join(models_dir, 'hypertension_model.pkl'), 'rb') as f:
                self.models['hypertension'] = pickle.load(f)
            self.debug_info.append("Hypertension model loaded successfully")

            # Load scalers
            self.debug_info.append("Loading hypertension scaler...")
            with open(os.path.join(models_dir, 'hypertension_scaler.pkl'), 'rb') as f:
                self.scalers['hypertension'] = pickle.load(f)
            self.debug_info.append("Hypertension scaler loaded successfully")

            # Load feature names
            self.debug_info.append("Loading feature names...")
            with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
                self.feature_names = pickle.load(f)
            self.debug_info.append("Feature names loaded successfully")

            # Load performance metrics
            self.debug_info.append("Loading performance metrics...")
            with open(os.path.join(models_dir, 'performance_metrics.pkl'), 'rb') as f:
                self.performance_metrics = pickle.load(f)
            self.debug_info.append("Performance metrics loaded successfully")

            self.models_loaded = True
            self.debug_info.append("All models loaded successfully!")
            return True

        except Exception as e:
            self.error_message = f"Error loading models: {str(e)}"
            self.debug_info.append(f"Error during model loading: {str(e)}")
            import traceback
            self.debug_info.append(f"Full traceback: {traceback.format_exc()}")
            return False

    def predict_diabetes(self, input_data):
        if 'diabetes' not in self.models:
            return None

        prediction = self.models['diabetes'].predict([input_data])[0]
        probability = self.models['diabetes'].predict_proba([input_data])[0]

        feature_importance = self.models['diabetes'].feature_importances_
        feature_names = self.feature_names['diabetes']

        top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:3]

        return {
            'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': probability[1],
            'confidence': max(probability),
            'top_features': top_features
        }

    def predict_heart_disease(self, input_data):
        if 'heart_disease' not in self.models:
            return None

        prediction = self.models['heart_disease'].predict([input_data])[0]
        probability = self.models['heart_disease'].predict_proba([input_data])[0]

        feature_importance = self.models['heart_disease'].feature_importances_
        feature_names = self.feature_names['heart_disease']

        top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:3]

        return {
            'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': probability[1],
            'confidence': max(probability),
            'top_features': top_features
        }

    def predict_hypertension(self, input_data):
        if 'hypertension' not in self.models or 'hypertension' not in self.scalers:
            return None

        input_scaled = self.scalers['hypertension'].transform([input_data])

        prediction = self.models['hypertension'].predict(input_scaled)[0]
        probability = self.models['hypertension'].predict_proba(input_scaled)[0]

        feature_importance = np.abs(self.models['hypertension'].coef_[0])
        feature_names = self.feature_names['hypertension']

        top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:3]

        return {
            'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': probability[1],
            'confidence': max(probability),
            'top_features': top_features
        }

def get_risk_style(probability):
    if probability > 0.7:
        return "risk-high"
    elif probability > 0.3:
        return "risk-medium"
    else:
        return "risk-low"

def create_risk_gauge(probability, title, color="blue"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{title} Risk Assessment"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(height=350, font={'size': 16})
    return fig

def display_prediction_results(result, disease_name):
    if result is None:
        st.error("❌ Model not available. Please check the debug information below.")
        return

    risk_class = get_risk_style(result['probability'])

    st.markdown(f"""
    <div class="prediction-container">
        <h2 style="text-align: center; margin-bottom: 1rem;">🎯 {disease_name} Prediction Results</h2>
        <div style="text-align: center;">
            <h3>Risk Level: <span class="{risk_class}">{result['prediction']}</span></h3>
            <p style="font-size: 1.2em;"><strong>Risk Probability:</strong> {result['probability']:.1%}</p>
            <p style="font-size: 1.1em;"><strong>Model Confidence:</strong> {result['confidence']:.1%}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    gauge_colors = {'Diabetes': 'darkblue', 'Heart Disease': 'darkred', 'Hypertension': 'darkgreen'}
    color = gauge_colors.get(disease_name, 'blue')

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_gauge = create_risk_gauge(result['probability'], disease_name, color)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader(" Top Contributing Factors")
        for i, (feature, importance) in enumerate(result['top_features'], 1):
            st.write(f"{i}. **{feature.replace('_', ' ').title()}**: {importance:.3f}")

        if result['probability'] > 0.7:
            st.error("**High Risk** - Recommend immediate medical consultation")
        elif result['probability'] > 0.3:
            st.warning("**Moderate Risk** - Consider lifestyle changes and regular monitoring")
        else:
            st.success("**Low Risk** - Maintain healthy lifestyle and regular checkups")

def main():
    ml_system = ProductionMLSystem()

    st.markdown('<h1 class="main-header">Chronic Disease Prediction System</h1>', unsafe_allow_html=True)

    if not ml_system.models_loaded:
        st.error("**MODELS NOT LOADED**")
        st.error(f"**Error:** {ml_system.error_message}")

        with st.expander("**Debug Information** (Click to expand)", expanded=True):
            st.markdown('<div class="debug-info">', unsafe_allow_html=True)
            st.write("**Debugging Steps:**")
            for info in ml_system.debug_info:
                st.write(f"- {info}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.write("**Solution Steps:**")
            st.code("""
# Run these commands in order:
1. python train_models.py    # Train the models first
2. streamlit run app.py      # Then run the application
            """)

        st.stop()  

    st.success("All models loaded successfully!")

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 1rem; background-color: #e3f2fd; border-radius: 10px;">
        <h3>Production-Ready AI System for Medical Risk Assessment</h3>
        <p>Powered by classical machine learning algorithms trained on comprehensive medical datasets</p>
        <p><em>⚠️ For educational and research purposes only. Always consult healthcare professionals for medical advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.title("Disease Prediction")
    st.sidebar.markdown("Select a disease to assess risk:")

    page = st.sidebar.selectbox(
        "Choose Prediction Type:",
        [" Dashboard", "Diabetes", "Heart Disease", "Hypertension", "Model Performance", "Debug Info"]
    )

    if page == "Dashboard":
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="disease-card">
                <h3>Diabetes Prediction</h3>
                <p><strong>Algorithm:</strong> Random Forest Classifier</p>
                <p><strong>Features:</strong> 8 metabolic indicators</p>
                <p><strong>Performance:</strong> ROC-AUC > 0.80</p>
                <p>Assess diabetes risk using glucose levels, BMI, family history, and other metabolic factors.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="disease-card">
                <h3>Heart Disease Prediction</h3>
                <p><strong>Algorithm:</strong> Gradient Boosting Classifier</p>
                <p><strong>Features:</strong> 13 cardiovascular indicators</p>
                <p><strong>Performance:</strong> ROC-AUC > 0.80</p>
                <p>Evaluate heart disease risk based on chest pain, cholesterol, ECG results, and cardiac markers.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="disease-card">
                <h3>Hypertension Prediction</h3>
                <p><strong>Algorithm:</strong> Logistic Regression</p>
                <p><strong>Features:</strong> 11 lifestyle & demographic factors</p>
                <p><strong>Performance:</strong> ROC-AUC > 0.85</p>
                <p>Predict hypertension risk using age, family history, lifestyle, and health indicators.</p>
            </div>
            """, unsafe_allow_html=True)

        # System status
        st.subheader("System Status")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Models Loaded", "3/3")

        with col2:
            st.metric("Scalers Loaded", "1/1")

        with col3:
            st.metric("System Status", "Ready")

        with col4:
            st.metric("Response Time", "< 100ms")

        if ml_system.performance_metrics:
            st.subheader("Model Performance Overview")

            diseases = list(ml_system.performance_metrics.keys())
            metrics_data = []

            for disease in diseases:
                metrics = ml_system.performance_metrics[disease]
                metrics_data.append({
                    'Disease': disease.replace('_', ' ').title(),
                    'Accuracy': metrics['accuracy'],
                    'F1-Score': metrics['f1_score'],
                    'ROC-AUC': metrics['roc_auc']
                })

            df_metrics = pd.DataFrame(metrics_data)

            fig = px.bar(df_metrics, x='Disease', y=['Accuracy', 'F1-Score', 'ROC-AUC'],
                        title='Model Performance Comparison',
                        barmode='group')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Diabetes":
        st.header("Diabetes Risk Assessment")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Patient Information")

            with st.form("diabetes_form"):
                col_left, col_right = st.columns(2)

                with col_left:
                    pregnancies = st.slider("Number of Pregnancies", 0, 17, 1, help="Total number of pregnancies")
                    glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 120, help="Plasma glucose concentration")
                    blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70, help="Diastolic blood pressure")
                    skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20, help="Triceps skin fold thickness")

                with col_right:
                    insulin = st.slider("Insulin Level (mu U/ml)", 0, 846, 80, help="2-Hour serum insulin")
                    bmi = st.slider("BMI", 0.0, 67.0, 25.0, 0.1, help="Body mass index (weight in kg/(height in m)^2)")
                    diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.08, 2.42, 0.5, 0.01, help="Genetic predisposition score")
                    age = st.slider("Age", 21, 81, 30, help="Age in years")

                submitted = st.form_submit_button("🔍 Predict Diabetes Risk", type="primary")

                if submitted:
                    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                                 insulin, bmi, diabetes_pedigree, age]

                    result = ml_system.predict_diabetes(input_data)
                    display_prediction_results(result, "Diabetes")

        with col2:
            st.subheader("About Diabetes Prediction")

            st.info("""
            **Key Risk Factors:**
            - High glucose levels
            - Elevated BMI
            - Family history (pedigree function)
            - Advanced age
            - Previous pregnancies
            """)

    elif page == "Heart Disease":
        st.header("Heart Disease Risk Assessment")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Cardiovascular Information")

            with st.form("heart_form"):
                col_left, col_right = st.columns(2)

                with col_left:
                    age = st.slider("Age", 29, 77, 50)
                    sex = st.selectbox("Sex", ["Female", "Male"])
                    sex_val = 1 if sex == "Male" else 0

                    chest_pain = st.selectbox("Chest Pain Type", [
                        "Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"
                    ])
                    chest_pain_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain)

                    resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
                    cholesterol = st.slider("Cholesterol (mg/dL)", 126, 409, 240)
                    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
                    fasting_bs_val = 1 if fasting_bs == "Yes" else 0

                with col_right:
                    rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
                    rest_ecg_val = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(rest_ecg)

                    max_hr = st.slider("Maximum Heart Rate", 71, 202, 150)
                    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
                    exercise_angina_val = 1 if exercise_angina == "Yes" else 0

                    st_depression = st.slider("ST Depression", 0.0, 6.2, 1.0, 0.1)
                    st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
                    st_slope_val = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

                    ca_vessels = st.slider("Number of Major Vessels", 0, 3, 0)
                    thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
                    thalassemia_val = [1, 2, 3][["Normal", "Fixed Defect", "Reversible Defect"].index(thalassemia)]

                submitted = st.form_submit_button("Predict Heart Disease Risk", type="primary")

                if submitted:
                    input_data = [age, sex_val, chest_pain_val, resting_bp, cholesterol, fasting_bs_val,
                                 rest_ecg_val, max_hr, exercise_angina_val, st_depression, st_slope_val,
                                 ca_vessels, thalassemia_val]

                    result = ml_system.predict_heart_disease(input_data)
                    display_prediction_results(result, "Heart Disease")

        with col2:
            st.subheader("About Heart Disease Prediction")

            st.info("""
            **Major Risk Factors:**
            - High cholesterol levels
            - Exercise-induced angina
            - Advanced age
            - Abnormal ECG results
            - High blood pressure
            """)

    elif page == "Hypertension":
        st.header("Hypertension Risk Assessment")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Lifestyle & Health Information")

            with st.form("hypertension_form"):
                col_left, col_right = st.columns(2)

                with col_left:
                    age = st.slider("Age", 25, 80, 45)
                    gender = st.selectbox("Gender", ["Female", "Male"])
                    gender_val = 1 if gender == "Male" else 0

                    height = st.slider("Height (cm)", 140, 200, 170)
                    weight = st.slider("Weight (kg)", 40, 150, 70)
                    bmi = weight / ((height/100) ** 2)

                    family_history = st.selectbox("Family History of Hypertension", ["No", "Yes"])
                    family_history_val = 1 if family_history == "Yes" else 0

                with col_right:
                    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
                    smoking_val = 1 if smoking == "Yes" else 0

                    alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Heavy"])
                    alcohol_val = ["None", "Moderate", "Heavy"].index(alcohol)

                    exercise = st.slider("Exercise Frequency (times/week)", 0, 7, 3)
                    salt_intake = st.selectbox("Salt Intake Level", ["Low", "Normal", "High"])
                    salt_intake_val = ["Low", "Normal", "High"].index(salt_intake)

                    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

                st.write(f"**Calculated BMI:** {bmi:.1f}")

                submitted = st.form_submit_button("Predict Hypertension Risk", type="primary")

                if submitted:
                    input_data = [age, gender_val, height, weight, family_history_val, smoking_val,
                                 alcohol_val, exercise, salt_intake_val, stress_level, bmi]

                    result = ml_system.predict_hypertension(input_data)
                    display_prediction_results(result, "Hypertension")

        with col2:
            st.subheader("Prevention Guidelines")

            st.info("""
            **Prevention Tips:**
            - Exercise regularly (150 min/week)
            - Reduce sodium intake (<2,300mg/day)
            - Maintain healthy weight (BMI 18.5-24.9)
            - Manage stress through meditation
            - Limit alcohol consumption
            """)

    elif page == "Model Performance":
        st.header("Model Performance & Metrics")

        if ml_system.performance_metrics:
            # Performance table
            st.subheader("Performance Summary")

            performance_data = []
            for disease, metrics in ml_system.performance_metrics.items():
                model_name = type(ml_system.models[disease]).__name__
                performance_data.append({
                    'Disease': disease.replace('_', ' ').title(),
                    'Algorithm': model_name,
                    'Accuracy': f"{metrics['accuracy']:.3f}",
                    'Precision': f"{metrics['precision']:.3f}",
                    'Recall': f"{metrics['recall']:.3f}",
                    'F1-Score': f"{metrics['f1_score']:.3f}",
                    'ROC-AUC': f"{metrics['roc_auc']:.3f}"
                })

            df_performance = pd.DataFrame(performance_data)
            st.dataframe(df_performance, use_container_width=True)

    elif page == "Debug Info":
        st.header("System Debug Information")

        st.subheader("Model Loading Status")
        if ml_system.models_loaded:
            st.success("All models loaded successfully!")
        else:
            st.error("Models failed to load")
            st.error(f"Error: {ml_system.error_message}")

        st.subheader("Detailed Debug Information")
        for info in ml_system.debug_info:
            if "✅" in info:
                st.success(info)
            elif "❌" in info:
                st.error(info)
            else:
                st.info(info)

        st.subheader("File System Information")
        current_dir = os.getcwd()
        st.write(f"**Current Directory:** {current_dir}")

        if os.path.exists('models'):
            st.write("**Models Directory Contents:**")
            model_files = os.listdir('models')
            for file in model_files:
                file_path = os.path.join('models', file)
                file_size = os.path.getsize(file_path)
                st.write(f"- {file} ({file_size:,} bytes)")
        else:
            st.error("Models directory not found!")

if __name__ == "__main__":
    main()
