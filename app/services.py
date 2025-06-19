# student-dropout-prediction-api/app/services.py
import pandas as pd
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from datetime import datetime # For timestamp in dashboard stats

# Import Pydantic models (specifically FeatureImportanceItem if it's defined in schemas.py)
from app.schemas import (
    StudentDataInput,
    PredictionResponse,
    StudentPredictionResult,
    FeatureImportanceItem, # Assuming this is in schemas.py for consistency
    AtRiskStats # New schema for dashboard stats
)
from app.core_utils import (
    get_predictor,
    create_feature_dataframe,
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT # Used for constructing data paths
)

from ml.ml_dropout_prediction_system import main_training_pipeline as run_ml_training
from sklearn.ensemble import VotingClassifier

# --- Define paths for dashboard sample data ---
DASHBOARD_SAMPLE_DATA_PATH = str(PROJECT_ROOT / "ml" / "data" / "dashboard_sample_students.csv") # Ideal sample
MAIN_DATASET_PATH = str(PROJECT_ROOT / "ml" / "data" / "student_dropout_dataset.csv")
GENERATED_DATASET_PATH = str(PROJECT_ROOT / "ml" / "data" / "generated_student_dropout_dataset.csv")


def predict_single_student(student_data: StudentDataInput) -> PredictionResponse:
    predictor = get_predictor()
    if not predictor or not predictor.is_fitted:
        raise ValueError("Model is not loaded or not fitted. Please train or load a model first.")

    student_data_dict = student_data.model_dump(exclude_none=False) 
    
    try:
        student_df = create_feature_dataframe(student_data_dict)
    except ValueError as e: 
        raise ValueError(f"Failed to prepare data for prediction: {e}")

    try:
        risk_assessment = predictor.predict_student_risk(student_df)
        return PredictionResponse(**risk_assessment)
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}. Ensure input data matches model expectations.")


def predict_batch_students(students_data: List[StudentDataInput]) -> List[StudentPredictionResult]:
    predictor = get_predictor()
    if not predictor or not predictor.is_fitted:
        raise ValueError("Model is not loaded or not fitted. Cannot perform batch prediction.")

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
                student_id=student_id,
                predicted_outcome="Error",
                probabilities={},
                dropout_probability=0.0,
                risk_level="Unknown",
                recommendations=[],
                error=str(e)
            ))
    return results

def train_model_pipeline() -> Dict[str, Any]:
    print("🚀 Starting ML model training pipeline...")
    try:
        trained_predictor, insights = run_ml_training()

        if trained_predictor and trained_predictor.best_model:
            from app.core_utils import load_predictor_model 
            load_predictor_model() 

            return {
                "status": "success",
                "message": "Training pipeline completed successfully. Model saved and reloaded.",
                "model_path": DEFAULT_MODEL_PATH, 
                "insights_generated": bool(insights)
            }
        else:
            return {
                "status": "failure",
                "message": "Training pipeline ran but model training might have failed.",
                 "model_path": None,
                 "insights_generated": False
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred during training: {str(e)}",
            "model_path": None,
            "insights_generated": False
        }

def get_model_information() -> Dict[str, Any]:
    predictor = get_predictor()
    info_for_schema = {
        "model_type": None, "training_date": None, "performance_metrics": None,
        "feature_importance_available": False, "model_ready": False,
    }
    internal_debug_info = { "message": "Model not loaded or not fully initialized." }

    if not predictor or not predictor.is_fitted or not predictor.best_model:
        return info_for_schema

    info_for_schema["model_ready"] = True
    internal_debug_info["message"] = "Model is loaded."
    if predictor.best_model:
        info_for_schema["model_type"] = predictor.best_model.__class__.__name__
    
    if predictor.feature_importance:
        # Check based on existence of permutation or direct importance for best model
        best_model_name = info_for_schema["model_type"]
        perm_key = f"{best_model_name}_Permutation" if best_model_name else None
        if isinstance(predictor.best_model, VotingClassifier): # Standardize ensemble key lookup
            perm_key = "Ensemble_Permutation"

        direct_key = f"{best_model_name}_Direct" if best_model_name else None
        if isinstance(predictor.best_model, VotingClassifier):
            direct_key = "Ensemble_Direct" # Though direct usually not for ensemble

        if (perm_key and perm_key in predictor.feature_importance and not predictor.feature_importance[perm_key].empty) or \
           (direct_key and direct_key in predictor.feature_importance and not predictor.feature_importance[direct_key].empty) or \
           (best_model_name and best_model_name in predictor.feature_importance and not predictor.feature_importance[best_model_name].empty):
            info_for_schema["feature_importance_available"] = True
        elif any(not df.empty for df in predictor.feature_importance.values() if isinstance(df, pd.DataFrame)): # General check
             info_for_schema["feature_importance_available"] = True


    best_model_name_in_metrics: Optional[str] = None 
    if predictor.best_model: 
        if isinstance(predictor.best_model, VotingClassifier) and \
           'Ensemble' in predictor.models and \
           predictor.models['Ensemble'] == predictor.best_model:
            best_model_name_in_metrics = 'Ensemble'
        else: 
            for name, model_obj in predictor.models.items():
                if model_obj == predictor.best_model:
                    best_model_name_in_metrics = name
                    break
    
    if best_model_name_in_metrics and \
       best_model_name_in_metrics in predictor.performance_metrics and \
       isinstance(predictor.performance_metrics[best_model_name_in_metrics], dict):
        info_for_schema["performance_metrics"] = predictor.performance_metrics[best_model_name_in_metrics]
    else:
        if predictor.best_model: 
             internal_debug_info["message"] += f" Performance metrics for '{best_model_name_in_metrics}' could not be retrieved/not dict."
    
    return info_for_schema

def get_feature_importance_data() -> Optional[Dict[str, Any]]:
    predictor = get_predictor()
    if not predictor or not predictor.is_fitted or not predictor.best_model or not predictor.feature_importance:
        return None

    importance_df: Optional[pd.DataFrame] = None
    model_name_for_importance: Optional[str] = None
    best_model_class_name = predictor.best_model.__class__.__name__
    
    # Prioritize Permutation Importance
    perm_key_ensemble = "Ensemble_Permutation"
    perm_key_model = f"{best_model_class_name}_Permutation"

    if isinstance(predictor.best_model, VotingClassifier) and perm_key_ensemble in predictor.feature_importance:
        importance_df = predictor.feature_importance[perm_key_ensemble]
        model_name_for_importance = "Ensemble (Permutation)"
    elif perm_key_model in predictor.feature_importance:
        importance_df = predictor.feature_importance[perm_key_model]
        model_name_for_importance = f"{best_model_class_name} (Permutation)"
    
    # Fallback to Direct Importance
    if importance_df is None or importance_df.empty:
        direct_key_ensemble = "Ensemble_Direct" # Less common
        direct_key_model = f"{best_model_class_name}_Direct"
        legacy_key_model = best_model_class_name # Original key before _Direct suffix

        if isinstance(predictor.best_model, VotingClassifier) and direct_key_ensemble in predictor.feature_importance:
            importance_df = predictor.feature_importance[direct_key_ensemble]
            model_name_for_importance = "Ensemble (Direct)"
        elif direct_key_model in predictor.feature_importance:
            importance_df = predictor.feature_importance[direct_key_model]
            model_name_for_importance = f"{best_model_class_name} (Direct)"
        elif legacy_key_model in predictor.feature_importance: # Fallback to original key naming
             importance_df = predictor.feature_importance[legacy_key_model]
             model_name_for_importance = f"{best_model_class_name} (Direct/Legacy)"


    if importance_df is None or not isinstance(importance_df, pd.DataFrame) or importance_df.empty:
        # Final fallback: any non-empty permutation DF
        for key, df_val in predictor.feature_importance.items():
            if "Permutation" in key and isinstance(df_val, pd.DataFrame) and not df_val.empty:
                importance_df = df_val
                model_name_for_importance = key.replace("_Permutation", " (Permutation - Fallback)")
                break
        if importance_df is None or importance_df.empty:
            print("Warning: No suitable feature importance DataFrame found.")
            return None

    if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
        print(f"Warning: Feature importance DataFrame for '{model_name_for_importance}' is missing 'feature' or 'importance' columns.")
        return None

    structured_importances: List[FeatureImportanceItem] = []
    for _index, row in importance_df.iterrows():
        structured_importances.append(
            FeatureImportanceItem(feature=str(row['feature']), importance=float(row['importance']))
        )
    
    return {
        "feature_importances": structured_importances,
        "model_type": model_name_for_importance if model_name_for_importance else "Unknown"
    }

# --- NEW SERVICE for Dashboard Statistics ---

def get_dashboard_risk_statistics() -> Optional[Dict[str, Any]]:
    predictor = get_predictor()
    if not predictor or not predictor.is_fitted or not predictor.preprocessor: # Check preprocessor too
        print("Dashboard Stats: Model or preprocessor not ready.")
        return None

    df_sample: Optional[pd.DataFrame] = None
    data_source_desc = "Unknown source"
    sample_size = 20 # Using your log's sample size for consistency

    source_path_to_try = None
    if Path(DASHBOARD_SAMPLE_DATA_PATH).exists():
        source_path_to_try = DASHBOARD_SAMPLE_DATA_PATH
        data_source_desc = f"Specific dashboard sample ({Path(DASHBOARD_SAMPLE_DATA_PATH).name})"
    elif Path(MAIN_DATASET_PATH).exists():
        source_path_to_try = MAIN_DATASET_PATH
    elif Path(GENERATED_DATASET_PATH).exists():
        source_path_to_try = GENERATED_DATASET_PATH

    if source_path_to_try:
        try:
            df_full_source = pd.read_csv(source_path_to_try)
            if not df_full_source.empty:
                df_sample = df_full_source.sample(n=min(sample_size, len(df_full_source)), random_state=42)
                if source_path_to_try != DASHBOARD_SAMPLE_DATA_PATH : # update desc if sampled
                     data_source_desc = f"Random sample of {len(df_sample)} from {Path(source_path_to_try).name}"
            else: # df_full_source is empty
                df_sample = pd.DataFrame() # Ensure df_sample is an empty DataFrame, not None
        except Exception as e:
            print(f"Dashboard Stats: Error loading or sampling from {source_path_to_try}: {e}")
            df_sample = pd.DataFrame() # Ensure df_sample is an empty DataFrame, not None
    
    if df_sample is None or df_sample.empty: # Check for None or empty DataFrame
        print("Dashboard Stats: No sample data successfully loaded/sampled to generate statistics.")
        return None

    # Prepare DataFrame for prediction: drop 'Target', handle 'Student_ID'
    df_features_original = df_sample.copy()
    if 'Target' in df_features_original.columns:
        df_features_original = df_features_original.drop(columns=['Target'])
    if 'Student_ID' in df_features_original.columns:
        df_features_original = df_features_original.drop(columns=['Student_ID'])

    # --- BATCH PREPROCESSING ---
    print(f"Dashboard Stats: Preprocessing {len(df_features_original)} students from sample for statistics (once)...")
    try:
        # The preprocess_data method in StudentDropoutPredictor expects original features.
        # It handles selecting the correct columns based on _all_trained_original_features.
        # We pass is_training=False. It returns (processed_data, None, None)
        df_processed_sample, _, _ = predictor.preprocess_data(df_features_original.copy(), is_training=False)
        print("Dashboard Stats: Sample preprocessing complete.")
    except Exception as e:
        print(f"Dashboard Stats: Error during batch preprocessing of sample: {e}")
        return None
    
    if df_processed_sample is None:
        print("Dashboard Stats: Batch preprocessing returned None.")
        return None
        
    # Now, df_processed_sample is a NumPy array (or sparse matrix)
    # We can iterate through its rows for prediction, directly using the model.

    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0
    dropout_count = 0
    enrolled_count = 0
    graduate_count = 0
    processed_count = 0

    model = predictor.best_model
    model_classes = model.classes_.tolist() if hasattr(model, 'classes_') else ['Dropout', 'Enrolled', 'Graduate'] # Fallback
    dropout_class_index = -1
    if 'Dropout' in model_classes:
        dropout_class_index = model_classes.index('Dropout')

    print(f"Dashboard Stats: Aggregating predictions for {df_processed_sample.shape[0]} processed students...")
    for i in range(df_processed_sample.shape[0]):
        try:
            instance_processed = df_processed_sample[i].reshape(1, -1) # Get i-th row and reshape for single prediction

            prediction_outcome = model.predict(instance_processed)[0]
            probabilities_raw = model.predict_proba(instance_processed)[0]
            
            dropout_prob = 0.0
            if dropout_class_index != -1 and dropout_class_index < len(probabilities_raw):
                dropout_prob = probabilities_raw[dropout_class_index]
            
            risk_level = 'High' if dropout_prob > 0.7 else ('Medium' if dropout_prob > 0.3 else 'Low')

            if risk_level == 'High': high_risk_count += 1
            elif risk_level == 'Medium': medium_risk_count += 1
            else: low_risk_count += 1 # Default to Low if not High or Medium

            if prediction_outcome == 'Dropout': dropout_count += 1
            elif prediction_outcome == 'Enrolled': enrolled_count += 1
            elif prediction_outcome == 'Graduate': graduate_count += 1
            
            processed_count +=1
        except Exception as e:
            print(f"Dashboard Stats: Error during direct model prediction for processed sample row {i}: {e}")

    if processed_count == 0 and df_processed_sample.shape[0] > 0:
        print("Dashboard Stats: All processed sample rows failed direct model prediction.")
        return None 

    return {
        "total_students_processed": processed_count,
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "low_risk_count": low_risk_count,
        "dropout_count": dropout_count,
        "enrolled_count": enrolled_count,
        "graduate_count": graduate_count,
        "data_source_timestamp": datetime.now().isoformat(),
        "data_source_description": data_source_desc
    }