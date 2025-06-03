# student-dropout-prediction-api/ml/ml_dropout_prediction_system.py

"""
Student Dropout Prediction and Academic Performance Classification System
Built by: ML Engineering Team
Purpose: Predict college dropout rates and classify student academic status

NOTE FOR API INTEGRATION:
- All file paths (e.g., for datasets, model saving/loading) are relative
  to the current working directory. Ensure the CWD is the project root
  (e.g., student-dropout-prediction-api/) when running any part of this system.
- The StudentDropoutPredictor class is designed to be imported and used by other services.
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Commented out as not strictly needed for API backend
# import seaborn as sns # Commented out as not strictly needed for API backend
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

class StudentDropoutPredictor:
    """
    Comprehensive ML system for student dropout prediction and academic performance classification
    """
    
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_importance = {}
        self.performance_metrics = {}
        self.best_model = None
        self.is_fitted = False # To track if preprocessor is fitted
        self._trained_numerical_features = [] # Store original numerical feature names
        self._trained_categorical_features = [] # Store original categorical feature names
        self._all_trained_original_features = [] # Store all original feature names used for training preprocessor
        
    def load_and_validate_dataset(self, file_path="ml/data/student_dropout_dataset.csv"):
        """Load and validate the existing dataset.
        Assumes file_path is relative to the project root or an absolute path.
        """
        try:
            print(f"📂 Loading dataset from: {file_path}")
            df = pd.read_csv(file_path)
            
            required_columns = ['Target'] 
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns for training: {missing_columns}")
            
            print(f"✅ Dataset validation successful!")
            print(f"   • Shape: {df.shape}")
            if 'Target' in df.columns:
                print(f"   • Target distribution:")
                target_counts = df['Target'].value_counts()
                for target_val, count in target_counts.items(): # Renamed to avoid conflict
                    percentage = (count / len(df)) * 100
                    print(f"     - {target_val}: {count:,} ({percentage:.1f}%)")
            
            print(f"   • Missing values: {df.isnull().sum().sum()}")
            print(f"   • Duplicate records: {df.duplicated().sum()}")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ Dataset file not found at: {file_path}")
            print("📋 Please ensure the dataset file exists at the specified path")
            print("🔄 Alternative: Use generate_dataset() method to create sample data")
            return None
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return None

    def generate_dataset(self, n_students=1000): # Added default n_students
        """Generate synthetic student dataset"""
        np.random.seed(42)
        data = []
        
        # Define categorical options
        genders = ['Male', 'Female', 'Other']
        marital_status_options = ['Single', 'Married', 'Divorced', 'Widowed'] # Renamed
        application_modes = ['Online', 'In-person', 'Phone', 'Mail']
        course_types = ['Daytime', 'Evening', 'Weekend', 'Online']
        previous_qualifications = ['High School', 'Vocational', 'Bachelor', 'Other']
        nationalities = ['Domestic', 'International_Developed', 'International_Developing']
        mothers_qualifications = ['No Education', 'Primary', 'Secondary', 'Higher Education', 'Unknown']
        fathers_qualifications = ['No Education', 'Primary', 'Secondary', 'Higher Education', 'Unknown']
        mothers_occupations = ['Unemployed', 'Unskilled', 'Skilled', 'Professional', 'Unknown']
        fathers_occupations = ['Unemployed', 'Unskilled', 'Skilled', 'Professional', 'Unknown']
        displaced_options = ['Yes', 'No']
        educational_special_needs = ['Yes', 'No']
        debtor_status = ['Yes', 'No']
        tuition_fees_status = ['Up to date', 'Late', 'Not paid']
        scholarship_holder = ['Yes', 'No']
        programs = ['Engineering', 'Business', 'Arts', 'Sciences', 'Medicine', 'Law', 'Education', 'Social Sciences']
        
        for i in range(n_students):
            student_id = f"STU_SYNTH_{i+1:05d}" 
            gender = np.random.choice(genders, p=[0.45, 0.52, 0.03])
            age_at_enrollment = max(17, min(35, int(np.random.normal(20, 3))))
            marital = np.random.choice(marital_status_options, p=[0.85, 0.12, 0.025, 0.005])
            application_mode = np.random.choice(application_modes, p=[0.6, 0.25, 0.1, 0.05])
            course = np.random.choice(course_types, p=[0.6, 0.2, 0.15, 0.05])
            program = np.random.choice(programs)
            prev_qual = np.random.choice(previous_qualifications, p=[0.7, 0.15, 0.1, 0.05])
            prev_qual_grade = max(40, min(100, np.random.normal(75, 15)))
            nationality = np.random.choice(nationalities, p=[0.7, 0.2, 0.1])
            mother_qual = np.random.choice(mothers_qualifications, p=[0.1, 0.25, 0.4, 0.2, 0.05])
            father_qual = np.random.choice(fathers_qualifications, p=[0.1, 0.25, 0.4, 0.2, 0.05])
            mother_occ = np.random.choice(mothers_occupations, p=[0.15, 0.3, 0.35, 0.15, 0.05])
            father_occ = np.random.choice(fathers_occupations, p=[0.1, 0.25, 0.4, 0.2, 0.05])
            displaced = np.random.choice(displaced_options, p=[0.05, 0.95])
            special_needs = np.random.choice(educational_special_needs, p=[0.08, 0.92])
            debtor = np.random.choice(debtor_status, p=[0.15, 0.85])
            tuition_status = np.random.choice(tuition_fees_status, p=[0.7, 0.2, 0.1])
            scholarship = np.random.choice(scholarship_holder, p=[0.3, 0.7])
            unemployment_rate = max(0, np.random.normal(8, 3))
            inflation_rate = np.random.normal(3, 1.5)
            gdp_growth = np.random.normal(2.5, 2)
            sem1_attendance = np.random.beta(7, 2) * 100
            sem1_gpa = max(0, min(4.0, np.random.normal(2.8, 0.8)))
            sem1_credited = np.random.randint(3, 8)
            sem1_enrolled = sem1_credited + np.random.randint(0, 3)
            sem1_evaluations = sem1_enrolled + np.random.randint(0, 2)
            sem1_approved = np.random.randint(max(0, sem1_credited-2), sem1_credited+1)
            sem1_grade = max(0, min(20, np.random.normal(13, 3)))
            sem1_without_evaluations = sem1_enrolled - sem1_evaluations
            sem2_attendance = max(0, min(100, sem1_attendance + np.random.normal(0, 10)))
            sem2_gpa = max(0, min(4.0, sem1_gpa + np.random.normal(0, 0.3)))
            sem2_credited = np.random.randint(3, 8)
            sem2_enrolled = sem2_credited + np.random.randint(0, 3)
            sem2_evaluations = sem2_enrolled + np.random.randint(0, 2)
            sem2_approved = np.random.randint(max(0, sem2_credited-2), sem2_credited+1)
            sem2_grade = max(0, min(20, sem1_grade + np.random.normal(0, 2)))
            sem2_without_evaluations = sem2_enrolled - sem2_evaluations
            
            dropout_prob = 0.1
            if sem1_gpa < 2.0: dropout_prob += 0.3
            elif sem1_gpa < 2.5: dropout_prob += 0.15
            if sem1_attendance < 70: dropout_prob += 0.25
            elif sem1_attendance < 80: dropout_prob += 0.1
            if tuition_status == 'Not paid': dropout_prob += 0.4
            elif tuition_status == 'Late': dropout_prob += 0.15
            if debtor == 'Yes': dropout_prob += 0.1
            if scholarship == 'No': dropout_prob += 0.05
            if displaced == 'Yes': dropout_prob += 0.15
            if special_needs == 'Yes': dropout_prob += 0.05
            if mother_qual in ['No Education', 'Primary']: dropout_prob += 0.05
            if father_qual in ['No Education', 'Primary']: dropout_prob += 0.05
            if age_at_enrollment > 25: dropout_prob += 0.1
            if unemployment_rate > 10: dropout_prob += 0.05
            
            dropout_prob = max(0, min(1, dropout_prob))
            is_dropout = np.random.random() < dropout_prob
            
            target_val = '' # Renamed
            if is_dropout:
                target_val = 'Dropout'
            else:
                avg_gpa = (sem1_gpa + sem2_gpa) / 2
                if avg_gpa >= 3.5:
                    target_val = 'Graduate'
                else:
                    target_val = 'Enrolled'
            
            if is_dropout and np.random.random() < 0.7:
                sem2_attendance, sem2_gpa, sem2_credited, sem2_enrolled, \
                sem2_evaluations, sem2_approved, sem2_grade, sem2_without_evaluations = [np.nan] * 8
            
            data.append({
                'Student_ID': student_id, 'Gender': gender, 'Age_at_enrollment': age_at_enrollment,
                'Marital_status': marital, 'Application_mode': application_mode, 'Course_type': course,
                'Academic_program': program, 'Previous_qualification': prev_qual,
                'Previous_qualification_grade': round(prev_qual_grade, 2), 'Nationality': nationality,
                'Mothers_qualification': mother_qual, 'Fathers_qualification': father_qual,
                'Mothers_occupation': mother_occ, 'Fathers_occupation': father_occ,
                'Displaced': displaced, 'Educational_special_needs': special_needs, 'Debtor': debtor,
                'Tuition_fees_up_to_date': tuition_status, 'Scholarship_holder': scholarship,
                'Unemployment_rate': round(unemployment_rate, 2), 'Inflation_rate': round(inflation_rate, 2),
                'GDP_growth': round(gdp_growth, 2), 'Sem1_attendance_rate': round(sem1_attendance, 2),
                'Sem1_GPA': round(sem1_gpa, 2), 'Sem1_curricular_units_credited': sem1_credited,
                'Sem1_curricular_units_enrolled': sem1_enrolled, 'Sem1_curricular_units_evaluations': sem1_evaluations,
                'Sem1_curricular_units_approved': sem1_approved, 'Sem1_grade': round(sem1_grade, 2),
                'Sem1_curricular_units_without_evaluations': sem1_without_evaluations,
                'Sem2_attendance_rate': round(sem2_attendance, 2) if not pd.isna(sem2_attendance) else np.nan,
                'Sem2_GPA': round(sem2_gpa, 2) if not pd.isna(sem2_gpa) else np.nan,
                'Sem2_curricular_units_credited': sem2_credited if not pd.isna(sem2_credited) else np.nan,
                'Sem2_curricular_units_enrolled': sem2_enrolled if not pd.isna(sem2_enrolled) else np.nan,
                'Sem2_curricular_units_evaluations': sem2_evaluations if not pd.isna(sem2_evaluations) else np.nan,
                'Sem2_curricular_units_approved': sem2_approved if not pd.isna(sem2_approved) else np.nan,
                'Sem2_grade': round(sem2_grade, 2) if not pd.isna(sem2_grade) else np.nan,
                'Sem2_curricular_units_without_evaluations': sem2_without_evaluations if not pd.isna(sem2_without_evaluations) else np.nan,
                'Target': target_val
            })
        return pd.DataFrame(data)
    
    def preprocess_data(self, df, is_training=True):
        """Comprehensive data preprocessing"""
        print("🔄 Starting data preprocessing...")
        
        # Work on a copy to avoid SettingWithCopyWarning if df is a slice
        df_processed = df.copy()

        if is_training:
            X = df_processed.drop(['Student_ID', 'Target'], axis=1, errors='ignore')
            y = df_processed['Target'] if 'Target' in df_processed.columns else None
            
            # Identify and store feature names for consistency
            self._trained_categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            self._trained_numerical_features = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
            self._all_trained_original_features = self._trained_numerical_features + self._trained_categorical_features
        else: # For prediction
            # Ensure input DataFrame has all columns the preprocessor expects, in any order
            # Missing columns will be problematic for ColumnTransformer unless handled explicitly
            # (e.g., by API ensuring all features from schema are present, possibly as NaN)
            if not self._all_trained_original_features:
                 raise RuntimeError("Preprocessor feature names not set. Train model first or load a model with this info.")
            
            # Select only the features the preprocessor was trained on
            # Add missing columns with NaN if not present in input df
            for col_name in self._all_trained_original_features:
                if col_name not in df_processed.columns:
                    df_processed[col_name] = np.nan # Imputer will handle this
            
            # Student_ID might not be present or needed for prediction features
            X = df_processed[self._all_trained_original_features] 
            y = None

        # Use stored feature lists for consistency
        numerical_features = self._trained_numerical_features if self._trained_numerical_features else X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
        categorical_features = self._trained_categorical_features if self._trained_categorical_features else X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"📊 Using {len(categorical_features)} categorical and {len(numerical_features)} numerical features for preprocessing.")
        
        if is_training or self.preprocessor is None:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ], remainder='passthrough' 
            )
            X_processed_data = self.preprocessor.fit_transform(X)
            self.is_fitted = True
            print(f"✅ Preprocessor fitted. Final feature count after OHE: {X_processed_data.shape[1]}")
        else:
            if not self.is_fitted:
                 raise RuntimeError("Preprocessor must be fitted before transforming data. Call with is_training=True or load a model.")
            X_processed_data = self.preprocessor.transform(X) # X here should only contain expected features
            print(f"✅ Preprocessing complete using existing preprocessor. Feature count: {X_processed_data.shape[1]}")

        feature_names_out_list = None # Renamed
        if is_training:
            try:
                feature_names_out_list = (self.preprocessor.named_transformers_['num']
                                    .get_feature_names_out(numerical_features).tolist() + 
                                 self.preprocessor.named_transformers_['cat']
                                    .named_steps['onehot'].get_feature_names_out(categorical_features).tolist())
                if self.preprocessor.remainder == 'passthrough' and X_processed_data.shape[1] > len(feature_names_out_list):
                    # This logic for remainder names needs careful checking if remainder is used.
                    # For now, assuming remainder is not significant or empty.
                    # If remainder columns exist, their names need to be derived from original X.
                    pass # Simplified, assuming no significant remainder features
            except AttributeError: 
                feature_names_out_list = numerical_features + list(self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
            
            return X_processed_data, y, feature_names_out_list
        else:
            return X_processed_data, y, None
    
    def train_models(self, X, y):
        """Train multiple ML models for comparison"""
        print("🚀 Training multiple ML models...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
            'SVM': SVC(random_state=42, probability=True),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500, early_stopping=True)
        }
        for name, model_instance in models_config.items(): # Renamed
            print(f"  🔧 Training {name}...")
            cv_scores = cross_val_score(model_instance, X_train, y_train, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='accuracy')
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            y_pred_proba = model_instance.predict_proba(X_test) if hasattr(model_instance, 'predict_proba') else None
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            
            self.models[name] = model_instance
            self.performance_metrics[name] = {
                'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(), 'test_accuracy': accuracy,
                'precision': precision, 'recall': recall, 'f1_score': f1,
                'predictions': y_pred.tolist(), # Store as list for JSON serialization if needed
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
                'classes': model_instance.classes_.tolist() if hasattr(model_instance, 'classes_') else None
            }
            print(f"    ✅ {name} - CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f}), Test Accuracy: {accuracy:.3f}")
        
        best_model_name = max(self.performance_metrics, key=lambda k: self.performance_metrics[k]['test_accuracy']) # Use .get for safety
        self.best_model = self.models[best_model_name]
        print(f"🏆 Best model selected: {best_model_name}")
        return X_train, X_test, y_train, y_test
    
    def analyze_feature_importance(self, feature_names):
        """Analyze feature importance for interpretation"""
        if not feature_names:
            print("⚠️ Cannot analyze feature importance without feature names (typically available after training).")
            return

        print("🔍 Analyzing feature importance...")
        for name, model_instance in self.models.items():
            if hasattr(model_instance, 'feature_importances_'):
                importances = model_instance.feature_importances_
                if len(feature_names) != len(importances):
                    print(f"  ⚠️ Mismatch in feature names ({len(feature_names)}) and importances ({len(importances)}) for {name}. Skipping.")
                    continue
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names, 'importance': importances
                }).sort_values('importance', ascending=False)
                self.feature_importance[name] = feature_importance_df
                print(f"  📈 Top 10 features for {name}:")
                print(feature_importance_df.head(10).to_string(index=False)) # Cleaner print
                print()
    
    def create_ensemble_model(self, X_train, y_train, X_test, y_test): # Added X_test, y_test
        """Create an ensemble model combining best performers AND evaluate it."""
        print("🎯 Creating and evaluating ensemble model...")
        
        if not self.models or not self.performance_metrics:
            print("❌ Models not trained yet. Cannot create ensemble.")
            return None

        sorted_models_by_perf = sorted(self.performance_metrics.items(), 
                           key=lambda x: x[1].get('test_accuracy', 0), reverse=True)
        
        top_prob_models = []
        for name, _ in sorted_models_by_perf:
            model_instance = self.models[name]
            if hasattr(model_instance, 'predict_proba'):
                top_prob_models.append((name, model_instance))
            if len(top_prob_models) >= 3: break
        
        if len(top_prob_models) < 2:
            print("⚠️ Not enough models support probability for soft voting ensemble. Skipping.")
            return None

        ensemble = VotingClassifier(estimators=top_prob_models, voting='soft')
        ensemble.fit(X_train, y_train)
        
        self.models['Ensemble'] = ensemble
        self.best_model = ensemble 
        print(f"  ✅ Ensemble created using: {[name for name, _ in top_prob_models]} and set as best_model.")

        # --- NEW: Evaluate Ensemble and store metrics ---
        print(f"  🔧 Evaluating Ensemble model...")
        y_pred_ensemble = ensemble.predict(X_test)
        y_pred_proba_ensemble = ensemble.predict_proba(X_test) if hasattr(ensemble, 'predict_proba') else None
        
        accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
        precision_ensemble, recall_ensemble, f1_ensemble, _ = precision_recall_fscore_support(
            y_test, y_pred_ensemble, average='weighted', zero_division=0
        )
        
        # Cross-validation for ensemble (optional, can be time-consuming)
        # For simplicity, we'll use the test set performance here.
        # If CV is desired:
        # cv_scores_ensemble = cross_val_score(ensemble, X_train, y_train, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='accuracy')
        # cv_mean_ensemble = cv_scores_ensemble.mean()
        # cv_std_ensemble = cv_scores_ensemble.std()
        # print(f"    📊 Ensemble CV Accuracy: {cv_mean_ensemble:.3f} (±{cv_std_ensemble:.3f})")
        
        self.performance_metrics['Ensemble'] = {
            'cv_mean': None, # Not calculated in this simplified version, can add later
            'cv_std': None,  # Not calculated
            'test_accuracy': accuracy_ensemble,
            'precision': precision_ensemble,
            'recall': recall_ensemble,
            'f1_score': f1_ensemble,
            'predictions': y_pred_ensemble.tolist(),
            'probabilities': y_pred_proba_ensemble.tolist() if y_pred_proba_ensemble is not None else None,
            'classes': ensemble.classes_.tolist() if hasattr(ensemble, 'classes_') else None
        }
        print(f"    ✅ Ensemble - Test Accuracy: {accuracy_ensemble:.3f}, F1: {f1_ensemble:.3f}")
        # --- END NEW ---
        return ensemble
    
    def generate_insights_report(self, df):
        """Generate comprehensive insights for educational institutions"""
        if 'Target' not in df.columns:
            print("⚠️ 'Target' column not found in DataFrame. Cannot generate insights.")
            return {}
        print("📊 Generating insights report...")
        insights = {'dataset_overview': {}, 'risk_factors': {}, 'demographic_patterns': {},
                    'academic_patterns': {}, 'financial_patterns': {}}
        
        df_copy = df.copy() # Work on a copy
        insights['dataset_overview'] = {
            'total_students': len(df_copy),
            'dropout_rate': (df_copy['Target'] == 'Dropout').mean() * 100 if 'Dropout' in df_copy['Target'].unique() else 0,
            'graduation_rate': (df_copy['Target'] == 'Graduate').mean() * 100 if 'Graduate' in df_copy['Target'].unique() else 0,
            'enrollment_rate': (df_copy['Target'] == 'Enrolled').mean() * 100 if 'Enrolled' in df_copy['Target'].unique() else 0
        }
        dropout_df = df_copy[df_copy['Target'] == 'Dropout']
        if not dropout_df.empty:
            insights['risk_factors']['low_gpa_dropout_percentage'] = (dropout_df['Sem1_GPA'] < 2.0).mean() * 100 if 'Sem1_GPA' in dropout_df else 'N/A'
            # ... other risk factors
        # ... other pattern calculations (ensure column existence checks)
        return insights # Simplified example
    
    def predict_student_risk(self, student_data_df):
        """Predict individual student risk level. Expects a DataFrame."""
        if self.best_model is None:
            raise ValueError("Model not trained or loaded. Please run train_models or load_model first.")
        if self.preprocessor is None or not self.is_fitted:
             raise ValueError("Preprocessor not available or not fitted. Please train or load model first.")
        
        if not isinstance(student_data_df, pd.DataFrame):
            try:
                student_data_df = pd.DataFrame([student_data_df]) 
            except Exception as e:
                raise ValueError(f"Input student_data must be a Pandas DataFrame or convertible to one. Error: {e}")

        # Ensure the input DataFrame has the columns the preprocessor expects.
        # This is critical. preprocess_data (is_training=False) will now handle this.
        student_processed, _, _ = self.preprocess_data(student_data_df.copy(), is_training=False)
        
        prediction = self.best_model.predict(student_processed)[0]
        probabilities = self.best_model.predict_proba(student_processed)[0]
        classes = self.best_model.classes_.tolist()
        dropout_prob = probabilities[classes.index('Dropout')] if 'Dropout' in classes else 0.0

        risk_assessment = {
            'predicted_outcome': prediction,
            'probabilities': dict(zip(classes, probabilities.round(4))), # Round for cleaner output
            'dropout_probability': round(dropout_prob, 4),
            'risk_level': 'High' if dropout_prob > 0.7 else ('Medium' if dropout_prob > 0.3 else 'Low'),
            'recommendations': self.generate_recommendations(student_data_df.iloc[0], dropout_prob)
        }
        return risk_assessment
    
    def generate_recommendations(self, student_series, dropout_probability):
        """Generate personalized recommendations for at-risk students"""
        recommendations = []
        # Using .get() for safety as student_series comes from input data
        if student_series.get('Sem1_GPA', 4.0) < 2.5:
            recommendations.append("📚 Academic Support: Consider tutoring or study groups to improve GPA.")
        if student_series.get('Sem1_attendance_rate', 100.0) < 80.0:
            recommendations.append("🎯 Attendance Improvement: Identify and address attendance barriers.")
        if student_series.get('Tuition_fees_up_to_date') != 'Up to date':
            recommendations.append("💰 Financial Counseling: Explore payment plans or financial aid.")
        # ... more recommendations based on other fields
        if dropout_probability > 0.5:
             recommendations.append("⚠️ High Dropout Risk: Prioritize engagement and support.")
        if not recommendations:
            recommendations.append("✅ No immediate intervention flags based on current data. Continue monitoring.")
        return recommendations
    
    def save_model(self, filepath="ml/models/student_dropout_model.pkl"):
        """Save the trained model and preprocessor.
        Assumes filepath is relative to project root or an absolute path.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'best_model': self.best_model,
            'preprocessor': self.preprocessor,
            'models': self.models, 
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted,
            '_trained_numerical_features': self._trained_numerical_features, # Save for consistency
            '_trained_categorical_features': self._trained_categorical_features,
            '_all_trained_original_features': self._all_trained_original_features
        }
        joblib.dump(model_data, filepath)
        print(f"💾 Model and preprocessor components saved to {filepath}")
    
    def load_model(self, filepath="ml/models/student_dropout_model.pkl"):
        """Load a trained model.
        Assumes filepath is relative to project root or an absolute path.
        """
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data.get('best_model')
            self.preprocessor = model_data.get('preprocessor')
            self.models = model_data.get('models', {}) 
            self.performance_metrics = model_data.get('performance_metrics', {})
            self.feature_importance = model_data.get('feature_importance', {})
            self.is_fitted = model_data.get('is_fitted', False)
            self._trained_numerical_features = model_data.get('_trained_numerical_features', [])
            self._trained_categorical_features = model_data.get('_trained_categorical_features', [])
            self._all_trained_original_features = model_data.get('_all_trained_original_features', [])
            
            if self.best_model is None or self.preprocessor is None or not self.is_fitted:
                print(f"⚠️ Model, preprocessor, or fit status invalid in {filepath}.")
                return False
            if not self._all_trained_original_features:
                 print(f"⚠️ Original feature list not found in model file. Prediction might be unreliable if input columns vary.")


            print(f"📂 Model and preprocessor components loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"❌ Model file not found at {filepath}.")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False

# --- Main training execution and utility functions ---
def main_training_pipeline():
    """Main execution function for training and evaluation"""
    print("🎓 STUDENT DROPOUT PREDICTION SYSTEM - TRAINING PIPELINE")
    print("=" * 60)
    
    predictor = StudentDropoutPredictor()
    
    # Path for dataset loading, relative to project root
    dataset_path = "ml/data/student_dropout_dataset.csv"
    generated_dataset_path = "ml/data/generated_student_dropout_dataset.csv"

    print("1️⃣ Loading student dataset...")
    df = predictor.load_and_validate_dataset(file_path=dataset_path)
    
    if df is None or df.empty:
        print(f"   🔄 Dataset not found at {dataset_path}. Falling back to sample dataset generation...")
        df = predictor.generate_dataset(n_students=2000) # Reduced for faster example
        if df is None or df.empty:
            print("❌ Failed to load or generate dataset. Exiting.")
            return None, None
        try:
            os.makedirs(os.path.dirname(generated_dataset_path), exist_ok=True)
            df.to_csv(generated_dataset_path, index=False)
            print(f"   💾 Generated dataset saved to {generated_dataset_path}")
        except Exception as e:
            print(f"   ⚠️ Could not save generated dataset: {e}")
    
    print(f"\n📋 Dataset Summary (using features from training data):")
    # Store original feature columns from the loaded/generated df for the sample prediction later
    # These are the columns *before* preprocessing, excluding Student_ID and Target
    if 'Target' in df.columns and 'Student_ID' in df.columns:
         training_df_feature_columns = df.drop(columns=['Target', 'Student_ID']).columns.tolist()
    elif 'Target' in df.columns:
         training_df_feature_columns = df.drop(columns=['Target']).columns.tolist()
    else: # Should not happen if validation passed or generation worked
         training_df_feature_columns = df.columns.tolist()


    print("\n2️⃣ Preprocessing data...")
    X, y, feature_names = predictor.preprocess_data(df, is_training=True)
    
    print("\n3️⃣ Training ML models...")
    X_train, X_test, y_train, y_test = predictor.train_models(X, y) # X_test, y_test are available here
    
    print("\n4️⃣ Analyzing feature importance...")
    predictor.analyze_feature_importance(feature_names)
    
    print("\n5️⃣ Creating ensemble model...")
    # Pass X_test, y_test to the modified create_ensemble_model method
    ensemble = predictor.create_ensemble_model(X_train, y_train, X_test, y_test) 
    if ensemble:
        print("   Ensemble model created, evaluated, and set as best_model.")
    else:
        print("   Ensemble model creation skipped or failed.")
    # ... (rest of the main_training_pipeline remains largely the same) ...
    # For brevity, I'll skip repeating the full printout sections for metrics and insights
    # as they are primarily for script output, not core logic adjustment.
    
    print("\n" + "=" * 50)
    print("📈 MODEL PERFORMANCE SUMMARY (Showing for best model if ensemble, else for all)")
    # ... (print summary logic) ...
    
    print("\n" + "=" * 50)
    print("🔮 SAMPLE STUDENT RISK ASSESSMENT")
    print("=" * 50)
    
    for model_name_iter, metrics_iter in predictor.performance_metrics.items(): # Renamed loop vars
        print(f"{model_name_iter}:")
        # Use .get for safety in case some metrics are None (like CV for ensemble initially)
        cv_mean_val = metrics_iter.get('cv_mean')
        cv_std_val = metrics_iter.get('cv_std')
        cv_output = f"{cv_mean_val:.3f} ± {cv_std_val:.3f}" if cv_mean_val is not None and cv_std_val is not None else "N/A"
        
        print(f"  • Cross-validation: {cv_output}")
        print(f"  • Test Accuracy: {metrics_iter.get('test_accuracy', 'N/A'):.3f}")
        print(f"  • Precision: {metrics_iter.get('precision', 'N/A'):.3f}")
        print(f"  • Recall: {metrics_iter.get('recall', 'N/A'):.3f}")
        print(f"  • F1-Score: {metrics_iter.get('f1_score', 'N/A'):.3f}")
        print()

    sample_student_dict_raw = { # Raw data, keys should match original df columns
        'Gender': 'Male', 'Age_at_enrollment': 23, 'Marital_status': 'Single', 
        'Application_mode': 'Online', 'Course_type': 'Evening', 'Academic_program': 'Engineering', 
        'Previous_qualification': 'High School', 'Previous_qualification_grade': 65.0, 
        'Nationality': 'Domestic', 'Mothers_qualification': 'Primary', 'Fathers_qualification': 'Secondary', 
        'Mothers_occupation': 'Unemployed', 'Fathers_occupation': 'Unskilled', 
        'Displaced': 'No', 'Educational_special_needs': 'No', 'Debtor': 'Yes', 
        'Tuition_fees_up_to_date': 'Late', 'Scholarship_holder': 'No', 
        'Unemployment_rate': 12.5, 'Inflation_rate': 4.2, 'GDP_growth': 1.1, 
        'Sem1_attendance_rate': 65.0, 'Sem1_GPA': 1.8, 'Sem1_curricular_units_credited': 4, 
        'Sem1_curricular_units_enrolled': 6, 'Sem1_curricular_units_evaluations': 7, 
        'Sem1_curricular_units_approved': 2, 'Sem1_grade': 9.5, 
        'Sem1_curricular_units_without_evaluations': 1, 
        'Sem2_attendance_rate': np.nan, 'Sem2_GPA': np.nan, 'Sem2_curricular_units_credited': np.nan, 
        'Sem2_curricular_units_enrolled': np.nan, 'Sem2_curricular_units_evaluations': np.nan, 
        'Sem2_curricular_units_approved': np.nan, 'Sem2_grade': np.nan, 
        'Sem2_curricular_units_without_evaluations': np.nan
    }
    
    # Create a DataFrame for prediction using only the features the model was trained on.
    # Use the stored _all_trained_original_features from the predictor instance
    # (which were set during predictor.preprocess_data with is_training=True)
    sample_student_for_prediction_dict = {}
    if predictor._all_trained_original_features:
        for col in predictor._all_trained_original_features:
            sample_student_for_prediction_dict[col] = sample_student_dict_raw.get(col, np.nan)
        sample_student_df = pd.DataFrame([sample_student_for_prediction_dict])
        
        try:
            risk_assessment = predictor.predict_student_risk(sample_student_df)
            print(f"👤 Student Profile: Example high-risk student")
            print(f"📊 Academic Status: GPA {sample_student_dict_raw.get('Sem1_GPA', 'N/A')}, Attendance {sample_student_dict_raw.get('Sem1_attendance_rate', 'N/A')}%")
            print(f"\n🎯 RISK ASSESSMENT:")
            print(f"  • Predicted Outcome: {risk_assessment['predicted_outcome']}")
            print(f"  • Dropout Probability: {risk_assessment['dropout_probability']:.1%}")
            print(f"  • Risk Level: {risk_assessment['risk_level']}")
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in risk_assessment['recommendations']: print(f"  • {rec}")
        except Exception as e:
            print(f"❌ Error during sample prediction: {e}")
    else:
        print("⚠️ Could not perform sample prediction as trained feature list is unavailable.")

    print("\n" + "=" * 50)
    print("💾 SAVING MODEL FOR DEPLOYMENT")
    # Default path relative to project root
    model_save_path = "ml/models/student_dropout_model.pkl"
    predictor.save_model(filepath=model_save_path)
    
    print("\n🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 50)
    
    return predictor, {} # Return empty dict for insights for now

class StudentRiskDashboard: # Example class, can be expanded
    def __init__(self, predictor_instance):
        self.predictor = predictor_instance
    
    def batch_risk_assessment(self, student_df_batch): # Renamed parameter
        results = []
        for idx, row_series in student_df_batch.iterrows(): # Renamed
            student_data_single_df = pd.DataFrame([row_series]) 
            try:
                risk = self.predictor.predict_student_risk(student_data_single_df)
                risk['student_id'] = row_series.get('Student_ID', f'Student_Batch_{idx}')
                results.append(risk)
            except Exception as e:
                print(f"Error predicting for student {row_series.get('Student_ID', f'Student_Batch_{idx}')}: {e}")
                results.append({'student_id': row_series.get('Student_ID', f'Student_Batch_{idx}'), 'error': str(e)})
        return pd.DataFrame(results)

def create_deployment_package_example():
    """Illustrates content for a deployment module. Not for direct execution here."""
    print("\n🚀 CREATING DEPLOYMENT PACKAGE EXAMPLE (Conceptual)")
    # The actual content of deployment_example.py is illustrative and was provided in previous steps.
    # This function just notes that such a file would be part of a deployment strategy.
    print("  ℹ️  A 'deployment_example.py' or similar API integration module would use the")
    print("     StudentDropoutPredictor class and its saved model ('student_dropout_model.pkl').")
    print("     The FastAPI app structure (app/main.py, app/services.py, etc.) serves this purpose.")

if __name__ == "__main__":
    trained_predictor, generated_insights = main_training_pipeline()
    
    if trained_predictor and trained_predictor.best_model:
        # Example of using the dashboard utility if needed
        # sample_test_df = trained_predictor.load_and_validate_dataset() # Reload or use a test set
        # if sample_test_df is not None and not sample_test_df.empty:
        #     dashboard = StudentRiskDashboard(trained_predictor)
        #     batch_results = dashboard.batch_risk_assessment(sample_test_df.head())
        #     print("\n📋 Sample Batch Risk Assessment from Dashboard:")
        #     print(batch_results)

        create_deployment_package_example()
        print("\n🎓 STUDENT DROPOUT ML SYSTEM PROCESSING COMPLETE!")
        print("   Model saved and ready for API integration. 🚀")
    else:
        print("\n❌ ML System training or initialization failed.")