import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class ChronicDiseaseModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.feature_names = {}

    def create_synthetic_datasets(self):
        print("Creating synthetic datasets...")

        # Set random seed for reproducibility
        np.random.seed(42)

        # 1. Diabetes Dataset (PIMA-style)
        n_diabetes = 768
        diabetes_data = {
            'pregnancies': np.random.randint(0, 17, n_diabetes),
            'glucose': np.clip(np.random.normal(120, 32, n_diabetes), 0, 200),
            'blood_pressure': np.clip(np.random.normal(69, 19, n_diabetes), 0, 122),
            'skin_thickness': np.clip(np.random.normal(20, 16, n_diabetes), 0, 99),
            'insulin': np.clip(np.random.normal(79, 115, n_diabetes), 0, 846),
            'bmi': np.clip(np.random.normal(32, 8, n_diabetes), 0, 67),
            'diabetes_pedigree': np.clip(np.random.normal(0.47, 0.33, n_diabetes), 0.08, 2.42),
            'age': np.random.randint(21, 81, n_diabetes)
        }

        # Create realistic diabetes outcome based on risk factors
        diabetes_risk = (
            (diabetes_data['glucose'] > 140) * 0.3 +
            (diabetes_data['bmi'] > 30) * 0.2 +
            (diabetes_data['age'] > 50) * 0.1 +
            (diabetes_data['diabetes_pedigree'] > 0.5) * 0.15 +
            np.random.normal(0, 0.2, n_diabetes)
        )
        diabetes_data['outcome'] = (diabetes_risk > 0.4).astype(int)

        diabetes_df = pd.DataFrame(diabetes_data)
        diabetes_df.to_csv('data/diabetes_dataset.csv', index=False)

        # 2. Heart Disease Dataset (UCI-style)
        n_heart = 303
        heart_data = {
            'age': np.random.randint(29, 77, n_heart),
            'sex': np.random.choice([0, 1], n_heart),
            'chest_pain_type': np.random.choice([0, 1, 2, 3], n_heart),
            'resting_bp': np.clip(np.random.normal(131, 17, n_heart), 94, 200),
            'cholesterol': np.clip(np.random.normal(246, 52, n_heart), 126, 409),
            'fasting_bs': np.random.choice([0, 1], n_heart, p=[0.85, 0.15]),
            'rest_ecg': np.random.choice([0, 1, 2], n_heart),
            'max_heart_rate': np.clip(np.random.normal(150, 23, n_heart), 71, 202),
            'exercise_angina': np.random.choice([0, 1], n_heart, p=[0.68, 0.32]),
            'st_depression': np.clip(np.random.normal(1.04, 1.16, n_heart), 0, 6.2),
            'st_slope': np.random.choice([0, 1, 2], n_heart),
            'ca_vessels': np.random.choice([0, 1, 2, 3], n_heart, p=[0.55, 0.22, 0.15, 0.08]),
            'thalassemia': np.random.choice([1, 2, 3], n_heart, p=[0.18, 0.16, 0.66])
        }

        heart_risk = (
            (heart_data['age'] > 55) * 0.15 +
            (heart_data['sex'] == 1) * 0.1 +
            (heart_data['chest_pain_type'] == 0) * 0.2 +
            (heart_data['cholesterol'] > 240) * 0.15 +
            (heart_data['exercise_angina'] == 1) * 0.2 +
            (heart_data['ca_vessels'] > 0) * 0.15 +
            np.random.normal(0, 0.15, n_heart)
        )
        heart_data['target'] = (heart_risk > 0.35).astype(int)

        heart_df = pd.DataFrame(heart_data)
        heart_df.to_csv('data/heart_disease_dataset.csv', index=False)

        # 3. Hypertension Dataset
        n_hyper = 500
        hypertension_data = {
            'age': np.random.randint(25, 80, n_hyper),
            'gender': np.random.choice([0, 1], n_hyper),
            'height': np.random.normal(165, 10, n_hyper),
            'weight': np.random.normal(70, 15, n_hyper),
            'family_history': np.random.choice([0, 1], n_hyper, p=[0.7, 0.3]),
            'smoking': np.random.choice([0, 1], n_hyper, p=[0.75, 0.25]),
            'alcohol': np.random.choice([0, 1, 2], n_hyper, p=[0.4, 0.4, 0.2]),
            'exercise': np.random.choice([0, 1, 2, 3], n_hyper, p=[0.2, 0.3, 0.35, 0.15]),
            'salt_intake': np.random.choice([0, 1, 2], n_hyper, p=[0.3, 0.5, 0.2]),
            'stress_level': np.random.randint(1, 11, n_hyper)
        }

        hypertension_data['bmi'] = hypertension_data['weight'] / ((hypertension_data['height']/100) ** 2)

        hypertension_risk = (
            (hypertension_data['age'] > 50) * 0.2 +
            (hypertension_data['bmi'] > 25) * 0.15 +
            (hypertension_data['family_history'] == 1) * 0.15 +
            (hypertension_data['smoking'] == 1) * 0.1 +
            (hypertension_data['salt_intake'] == 2) * 0.1 +
            (hypertension_data['stress_level'] > 7) * 0.1 +
            (hypertension_data['exercise'] == 0) * 0.1 +
            np.random.normal(0, 0.15, n_hyper)
        )
        hypertension_data['hypertension'] = (hypertension_risk > 0.4).astype(int)

        hypertension_df = pd.DataFrame(hypertension_data)
        hypertension_df.to_csv('data/hypertension_dataset.csv', index=False)

        print(f"Diabetes dataset: {diabetes_df.shape}")
        print(f"Heart disease dataset: {heart_df.shape}")
        print(f"Hypertension dataset: {hypertension_df.shape}")

        return diabetes_df, heart_df, hypertension_df

    def load_or_create_datasets(self):
        try:
            diabetes_df = pd.read_csv('data/diabetes_dataset.csv')
            heart_df = pd.read_csv('data/heart_disease_dataset.csv')
            hypertension_df = pd.read_csv('data/hypertension_dataset.csv')
            print("✓ Loaded existing datasets from data/ directory")
            return diabetes_df, heart_df, hypertension_df
        except FileNotFoundError:
            print("Dataset files not found. Creating synthetic datasets...")
            return self.create_synthetic_datasets()

    def train_diabetes_model(self, df):
        print("\nTraining Diabetes Prediction Model...")
        print("-" * 40)

        X = df.drop('outcome', axis=1)
        y = df['outcome']

        self.feature_names['diabetes'] = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        rf = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Evaluate model
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        self.models['diabetes'] = best_model
        self.performance_metrics['diabetes'] = metrics

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

        return best_model, metrics

    def train_heart_disease_model(self, df):
        print("\nTraining Heart Disease Prediction Model...")
        print("-" * 45)

        # Prepare data
        X = df.drop('target', axis=1)
        y = df['target']

        # Store feature names
        self.feature_names['heart_disease'] = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        gb = GradientBoostingClassifier(random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }

        grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Evaluate model
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        self.models['heart_disease'] = best_model
        self.performance_metrics['heart_disease'] = metrics

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

        return best_model, metrics

    def train_hypertension_model(self, df):
        print("\nTraining Hypertension Prediction Model...")
        print("-" * 42)

        # Prepare data
        X = df.drop('hypertension', axis=1)
        y = df['hypertension']

        # Store feature names
        self.feature_names['hypertension'] = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(random_state=42, max_iter=1000)

        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        self.models['hypertension'] = best_model
        self.scalers['hypertension'] = scaler
        self.performance_metrics['hypertension'] = metrics

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

        return best_model, scaler, metrics

    def save_models(self):
        os.makedirs('models', exist_ok=True)

        for name, model in self.models.items():
            with open(f'models/{name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} model")

        for name, scaler in self.scalers.items():
            with open(f'models/{name}_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Saved {name} scaler")

        with open('models/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(" Saved feature names")

        with open('models/performance_metrics.pkl', 'wb') as f:
            pickle.dump(self.performance_metrics, f)
        print("Saved performance metrics")

    def generate_model_report(self):

        report = "# Chronic Disease Prediction Models - Performance Report\n\n"
        report += "## Model Training Summary\n\n"

        for disease, metrics in self.performance_metrics.items():
            report += f"### {disease.replace('_', ' ').title()} Model\n"
            report += f"- **Algorithm**: {type(self.models[disease]).__name__}\n"
            report += f"- **Accuracy**: {metrics['accuracy']:.3f}\n"
            report += f"- **Precision**: {metrics['precision']:.3f}\n"
            report += f"- **Recall**: {metrics['recall']:.3f}\n"
            report += f"- **F1-Score**: {metrics['f1_score']:.3f}\n"
            report += f"- **ROC-AUC**: {metrics['roc_auc']:.3f}\n\n"

        report += "## Feature Importance\n\n"

        for disease, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                feature_names = self.feature_names[disease]
                importances = model.feature_importances_

                # Get top 5 features
                top_indices = np.argsort(importances)[-5:][::-1]

                report += f"### {disease.replace('_', ' ').title()} - Top 5 Features\n"
                for i, idx in enumerate(top_indices, 1):
                    report += f"{i}. **{feature_names[idx]}**: {importances[idx]:.3f}\n"
                report += "\n"

        report += "## Model Files\n\n"
        report += "The following files have been generated:\n"
        report += "- `models/diabetes_model.pkl` - Diabetes prediction model\n"
        report += "- `models/heart_disease_model.pkl` - Heart disease prediction model\n"
        report += "- `models/hypertension_model.pkl` - Hypertension prediction model\n"
        report += "- `models/hypertension_scaler.pkl` - Feature scaler for hypertension model\n"
        report += "- `models/feature_names.pkl` - Feature names for all models\n"
        report += "- `models/performance_metrics.pkl` - Performance metrics for all models\n"

        with open('MODEL_REPORT.md', 'w') as f:
            f.write(report)


    def train_all_models(self):
        print("Starting Model Training")
        print("=" * 40)

        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        diabetes_df, heart_df, hypertension_df = self.load_or_create_datasets()

        self.train_diabetes_model(diabetes_df)
        self.train_heart_disease_model(heart_df)
        self.train_hypertension_model(hypertension_df)

        self.save_models()

        self.generate_model_report()

        print("\n" + "=" * 40)
        print("Models are saved and can be used now")

        return self.models, self.scalers, self.performance_metrics

if __name__ == "__main__":
    trainer = ChronicDiseaseModelTrainer()
    models, scalers, metrics = trainer.train_all_models()
