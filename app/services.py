# student-dropout-prediction-api/app/services.py
import pandas as pd
from typing import List, Dict, Any, Optional # Import Optional
import os
from pathlib import Path

from app.schemas import StudentDataInput, PredictionResponse, StudentPredictionResult
from app.core_utils import get_predictor, create_feature_dataframe, DEFAULT_MODEL_PATH, DEFAULT_DATA_DIR, PROJECT_ROOT

# Import the training pipeline function
# Ensure sys.path modification in core_utils has taken effect
from ml.ml_dropout_prediction_system import main_training_pipeline as run_ml_training
from sklearn.ensemble import VotingClassifier # Added for type checking

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
        # import logging; logging.exception("Prediction error") # For server-side debugging
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
    """
    Triggers the main training pipeline from ml_dropout_prediction_system.py
    """
    print("🚀 Starting ML model training pipeline...")
    try:
        trained_predictor, insights = run_ml_training()

        if trained_predictor and trained_predictor.best_model:
            from app.core_utils import load_predictor_model # To refresh global instance
            load_predictor_model() 

            return {
                "status": "success",
                "message": "Training pipeline completed successfully. Model saved and reloaded.",
                "model_path": DEFAULT_MODEL_PATH, 
                "insights_generated": bool(insights) # True if insights dict is not empty
            }
        else:
            return {
                "status": "failure",
                "message": "Training pipeline ran but model training might have failed.",
                 "model_path": None,
                 "insights_generated": False
            }
    except Exception as e:
        # import logging; logging.exception("Training error") # For server-side debugging
        return {
            "status": "error",
            "message": f"An error occurred during training: {str(e)}",
            "model_path": None,
            "insights_generated": False
        }

def get_model_information() -> Dict[str, Any]:
    predictor = get_predictor()

    # Initialize with defaults that match ModelInfoResponse schema fields
    info_for_schema = {
        "model_type": None,
        "training_date": None, 
        "performance_metrics": None,
        "feature_importance_available": False,
        "model_ready": False,
    }
    
    # Additional internal info, not directly part of ModelInfoResponse schema
    # but can be useful for constructing the final dict or for debugging.
    internal_debug_info = {
         "message": "Model not loaded or not fully initialized.", 
         "loaded_model_path": None,
         "preprocessor_fitted": False,
         "available_models_metrics": None
    }


    if not predictor or not predictor.is_fitted or not predictor.best_model:
        # Return only the schema-compatible fields for the "not ready" state
        return info_for_schema

    # If predictor and model seem to be loaded and fitted
    info_for_schema["model_ready"] = True
    internal_debug_info["preprocessor_fitted"] = predictor.is_fitted
    # Attempt to get the path it was loaded from (relies on how load_model was called)
    # This is a bit of a guess; for robustness, store loaded_path in predictor instance if needed
    internal_debug_info["loaded_model_path"] = DEFAULT_MODEL_PATH 
    internal_debug_info["available_models_metrics"] = predictor.performance_metrics
    internal_debug_info["message"] = "Model is loaded."


    if predictor.best_model:
        info_for_schema["model_type"] = predictor.best_model.__class__.__name__
    
    # Check for feature importance availability
    if predictor.feature_importance:
        best_model_type_name = info_for_schema["model_type"] # Can be None if best_model is None
        
        # Check if there's importance data specifically for the best model
        if best_model_type_name and best_model_type_name in predictor.feature_importance:
            info_for_schema["feature_importance_available"] = True
        elif best_model_type_name == "VotingClassifier" and 'Ensemble' in predictor.feature_importance:
            # Special case if ensemble feature importance is stored under 'Ensemble' key
            info_for_schema["feature_importance_available"] = True
        elif predictor.feature_importance: # First check if the dict itself is not empty
            # Check if any of the DataFrame values in the dictionary are non-empty
            is_any_df_non_empty = False
            for df_importance in predictor.feature_importance.values():
                if isinstance(df_importance, pd.DataFrame) and not df_importance.empty:
                    is_any_df_non_empty = True
                    break
            if is_any_df_non_empty:
                info_for_schema["feature_importance_available"] = True


    # Determine the key for the best model's metrics from predictor.performance_metrics
    best_model_name_in_metrics: Optional[str] = None 
    
    if predictor.best_model: # Ensure best_model exists before checking its type
        if isinstance(predictor.best_model, VotingClassifier) and \
           'Ensemble' in predictor.models and \
           predictor.models['Ensemble'] == predictor.best_model:
            best_model_name_in_metrics = 'Ensemble'
        else: # For single models
            for name, model_obj in predictor.models.items():
                if model_obj == predictor.best_model:
                    best_model_name_in_metrics = name
                    break
    
    if best_model_name_in_metrics and \
       best_model_name_in_metrics in predictor.performance_metrics and \
       isinstance(predictor.performance_metrics[best_model_name_in_metrics], dict):
        info_for_schema["performance_metrics"] = predictor.performance_metrics[best_model_name_in_metrics]
    else:
        # performance_metrics remains None (its default) if not found or not a dict
        if predictor.best_model: 
             internal_debug_info["message"] += " Performance metrics for the specific best model name could not be retrieved or were not in the expected dictionary format."
    
    # The function is typed to return Dict[str, Any].
    # The caller (app/main.py) will then use this dict to populate ModelInfoResponse.
    # Pydantic will take care of picking the fields it needs.
    # We can return a merged dict or just info_for_schema if we are sure it contains all Pydantic needs.
    # For clarity and to match the intended use with ModelInfoResponse, let's ensure
    # the returned dict primarily contains what ModelInfoResponse expects.
    
    # If you want to log internal_debug_info, do it here.
    # print(f"Internal debug info for /model-info: {internal_debug_info}")

    return info_for_schema