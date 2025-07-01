# student-dropout-prediction-api/app/services.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from datetime import datetime
import collections

# Import Pydantic models
from app.schemas import (
    StudentDataInput,
    PredictionResponse,
    StudentPredictionResult,
    FeatureImportanceItem,
    AtRiskStats
)
from app.core_utils import (
    get_predictor,
    create_feature_dataframe,
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT
)

from ml.ml_dropout_prediction_system import main_training_pipeline as run_ml_training
from sklearn.ensemble import VotingClassifier

# --- Define paths for dashboard sample data ---
DASHBOARD_SAMPLE_DATA_PATH = str(PROJECT_ROOT / "ml" / "data" / "dashboard_sample_students.csv")
MAIN_DATASET_PATH = str(PROJECT_ROOT / "ml" / "data" / "student_dropout_dataset.csv")
GENERATED_DATASET_PATH = str(PROJECT_ROOT / "ml" / "data" / "generated_student_dropout_dataset.csv")

# --- Functions that are already correct and do not need changes ---
def predict_single_student(student_data: StudentDataInput) -> PredictionResponse:
    # This function is fine as predict_student_risk in the ML script should handle the label mapping.
    # We will assume or verify that predict_student_risk returns string labels.
    predictor = get_predictor()
    if not predictor or not predictor.is_fitted:
        raise ValueError("Model is not loaded or not fitted.")
    student_data_dict = student_data.model_dump(exclude_none=False) 
    try:
        student_df = create_feature_dataframe(student_data_dict)
    except ValueError as e: 
        raise ValueError(f"Failed to prepare data for prediction: {e}")
    try:
        risk_assessment = predictor.predict_student_risk(student_df)
        return PredictionResponse(**risk_assessment)
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}.")

def predict_batch_students(students_data: List[StudentDataInput]) -> List[StudentPredictionResult]:
    # This function is fine.
    predictor = get_predictor()
    if not predictor or not predictor.is_fitted:
        raise ValueError("Model is not loaded or not fitted.")
    results = []
    for i, student_data in enumerate(students_data):
        student_id = f"student_input_{i+1}" 
        try:
            student_data_dict = student_data.model_dump(exclude_none=False)
            student_df = create_feature_dataframe(student_data_dict)
            risk_assessment = predictor.predict_student_risk(student_df)
            result = StudentPredictionResult(student_id=student_id, **risk_assessment)
            results.append(result)
        except Exception as e:
            results.append(StudentPredictionResult(
                student_id=student_id, predicted_outcome="Error", probabilities={},
                dropout_probability=0.0, risk_level="Unknown", recommendations=[], error=str(e)
            ))
    return results

def train_model_pipeline() -> Dict[str, Any]:
    # This function is fine.
    print("🚀 Starting ML model training pipeline...")
    try:
        trained_predictor, insights = run_ml_training()
        if trained_predictor and trained_predictor.best_model:
            from app.core_utils import load_predictor_model 
            load_predictor_model() 
            return {"status": "success", "message": "Training pipeline completed successfully.", "model_path": DEFAULT_MODEL_PATH, "insights_generated": bool(insights)}
        else:
            return {"status": "failure", "message": "Training pipeline ran but model training might have failed.", "model_path": None, "insights_generated": False}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred during training: {str(e)}", "model_path": None, "insights_generated": False}

def get_model_information() -> Dict[str, Any]:
    # This function is fine.
    predictor = get_predictor()
    info_for_schema = {"model_type": None, "training_date": None, "performance_metrics": None, "feature_importance_available": False, "model_ready": False}
    if not predictor or not predictor.is_fitted or not predictor.best_model:
        return info_for_schema
    info_for_schema["model_ready"] = True
    info_for_schema["model_type"] = predictor.best_model.__class__.__name__
    if predictor.feature_importance and any(not df.empty for df in predictor.feature_importance.values() if isinstance(df, pd.DataFrame)):
        info_for_schema["feature_importance_available"] = True
    best_model_name_in_metrics = 'Ensemble' if isinstance(predictor.best_model, VotingClassifier) and 'Ensemble' in predictor.models else predictor.best_model.__class__.__name__
    if best_model_name_in_metrics in predictor.performance_metrics:
        info_for_schema["performance_metrics"] = predictor.performance_metrics[best_model_name_in_metrics]
    return info_for_schema

def get_feature_importance_data() -> Optional[Dict[str, Any]]:
    # This function is fine.
    predictor = get_predictor()
    if not predictor or not predictor.is_fitted or not predictor.best_model or not predictor.feature_importance:
        return None
    importance_df, model_name_for_importance = None, None
    best_model_class_name = predictor.best_model.__class__.__name__
    perm_key = f"Ensemble_Permutation" if isinstance(predictor.best_model, VotingClassifier) else f"{best_model_class_name}_Permutation"
    direct_key = f"Ensemble_Direct" if isinstance(predictor.best_model, VotingClassifier) else f"{best_model_class_name}_Direct"
    if perm_key in predictor.feature_importance:
        importance_df, model_name_for_importance = predictor.feature_importance[perm_key], f"{best_model_class_name} (Permutation)"
    elif direct_key in predictor.feature_importance:
        importance_df, model_name_for_importance = predictor.feature_importance[direct_key], f"{best_model_class_name} (Direct)"
    if importance_df is None or importance_df.empty: return None
    structured_importances = [FeatureImportanceItem(feature=str(row['feature']), importance=float(row['importance'])) for _, row in importance_df.iterrows()]
    return {"feature_importances": structured_importances, "model_type": model_name_for_importance}


# --- FINAL, ROBUST, AND CORRECTED FUNCTION for Dashboard Statistics ---
def get_dashboard_risk_statistics() -> Optional[Dict[str, Any]]:
    """
    Generates risk statistics by running predictions on a sample dataset efficiently and correctly,
    handling numerical class labels from the model.
    """
    predictor = get_predictor()
    if not predictor or not predictor.is_fitted or not predictor.preprocessor:
        print("Dashboard Stats: Model or preprocessor not ready.")
        return None

    df_sample: Optional[pd.DataFrame] = None
    data_source_desc = "Unknown source"
    sample_size = 200

    source_path_to_try = None
    if Path(DASHBOARD_SAMPLE_DATA_PATH).exists():
        source_path_to_try = DASHBOARD_SAMPLE_DATA_PATH
    elif Path(MAIN_DATASET_PATH).exists():
        source_path_to_try = MAIN_DATASET_PATH
    elif Path(GENERATED_DATASET_PATH).exists():
        source_path_to_try = GENERATED_DATASET_PATH

    if source_path_to_try:
        try:
            df_full_source = pd.read_csv(source_path_to_try)
            if not df_full_source.empty:
                df_sample = df_full_source.sample(n=min(sample_size, len(df_full_source)), random_state=42)
                data_source_desc = f"Random sample of {len(df_sample)} from {Path(source_path_to_try).name}"
        except Exception as e:
            print(f"Dashboard Stats: Error loading or sampling from {source_path_to_try}: {e}")
            df_sample = pd.DataFrame()
    
    if df_sample is None or df_sample.empty:
        print("Dashboard Stats: No sample data successfully loaded to generate statistics.")
        return None

    df_features_original = df_sample.drop(columns=['Target', 'Student_ID'], errors='ignore')

    print(f"Dashboard Stats: Preprocessing a batch of {len(df_features_original)} students...")
    try:
        df_processed_sample, _, _ = predictor.preprocess_data(df_features_original, is_training=False)
        print("Dashboard Stats: Batch preprocessing complete.")
    except Exception as e:
        print(f"Dashboard Stats: Error during batch preprocessing of sample: {e}")
        return None
        
    if df_processed_sample is None: return None
        
    print(f"Dashboard Stats: Running batch prediction on {df_processed_sample.shape[0]} processed students...")
    model = predictor.best_model
    predicted_outcomes_numeric = model.predict(df_processed_sample)
    predicted_probabilities = model.predict_proba(df_processed_sample)
    
    # --- ROBUST MAPPING AND COUNTING FOR NUMERICAL LABELS ---
    try:
        model_classes_numeric = model.classes_.tolist()
        print(f"DEBUG: Model classes detected are: {model_classes_numeric}")

        # ** THIS IS THE CRITICAL FIX **
        # Define the mapping from the numeric code to the string label.
        # This order should match how the LabelEncoder was fitted in your ML script.
        # A common default is alphabetical: ['Dropout', 'Enrolled', 'Graduate']
        class_mapping = {
            0: 'Dropout',
            1: 'Enrolled',
            2: 'Graduate'
        }
        
        # Verify the mapping covers all detected classes
        if not all(c in class_mapping for c in model_classes_numeric):
            print(f"CRITICAL WARNING: The class_mapping in services.py is missing keys for detected model classes {model_classes_numeric}.")
            # Fallback to just using numbers as strings if mapping is incomplete
            class_mapping = {c: str(c) for c in model_classes_numeric}

        # Map the numeric predictions to string labels
        predicted_outcomes_string = [class_mapping.get(p, 'Unknown') for p in predicted_outcomes_numeric]

        # Find the numeric code for 'Dropout' to get the correct probability column
        dropout_numeric_label = next((k for k, v in class_mapping.items() if v == 'Dropout'), -1)

    except (ValueError, AttributeError):
        print("Warning: Could not read model.classes_. Predictions may be incorrect.")
        return None

    # Calculate risk levels for the entire batch
    if dropout_numeric_label != -1:
        # Use the numeric label as the index into the probability array
        dropout_probs = predicted_probabilities[:, dropout_numeric_label]
        risk_levels = np.select(
            [dropout_probs > 0.7, dropout_probs > 0.3], # Conditions
            ['High', 'Medium'],                         # Choices
            default='Low'                               # Default
        )
    else:
        risk_levels = ['Unknown'] * len(predicted_outcomes_string)

    # Use collections.Counter for robust counting of both risks and string outcomes
    risk_counts = collections.Counter(risk_levels)
    outcome_counts = collections.Counter(predicted_outcomes_string)

    print("Dashboard Stats: Aggregation complete.")
    
    return {
        "total_students_processed": len(predicted_outcomes_string),
        "high_risk_count": risk_counts.get('High', 0),
        "medium_risk_count": risk_counts.get('Medium', 0),
        "low_risk_count": risk_counts.get('Low', 0),
        "dropout_count": outcome_counts.get('Dropout', 0),
        "enrolled_count": outcome_counts.get('Enrolled', 0),
        "graduate_count": outcome_counts.get('Graduate', 0),
        "data_source_timestamp": datetime.now().isoformat(),
        "data_source_description": data_source_desc
    }