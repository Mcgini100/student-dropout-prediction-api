# student-dropout-prediction-api/app/services.py
import pandas as pd
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

# Import Pydantic models (specifically FeatureImportanceItem if it's defined in schemas.py)
from app.schemas import (
    StudentDataInput,
    PredictionResponse,
    StudentPredictionResult,
    FeatureImportanceItem # Assuming this is in schemas.py for consistency
)
from app.core_utils import get_predictor, create_feature_dataframe, DEFAULT_MODEL_PATH, DEFAULT_DATA_DIR, PROJECT_ROOT

from ml.ml_dropout_prediction_system import main_training_pipeline as run_ml_training
from sklearn.ensemble import VotingClassifier # For type checking

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
        # Pydantic will validate risk_assessment against PredictionResponse schema
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
                probabilities={}, # Ensure this matches PredictionResponse schema
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
        # import logging; logging.exception("Training error") # For server-side debugging
        return {
            "status": "error",
            "message": f"An error occurred during training: {str(e)}",
            "model_path": None,
            "insights_generated": False
        }

def get_model_information() -> Dict[str, Any]:
    predictor = get_predictor()

    info_for_schema = {
        "model_type": None,
        "training_date": None, 
        "performance_metrics": None,
        "feature_importance_available": False, # Will be updated if importance is found
        "model_ready": False,
    }
    
    internal_debug_info = {
         "message": "Model not loaded or not fully initialized.", 
         "loaded_model_path": None,
         "preprocessor_fitted": False,
         "available_models_metrics": None
    }

    if not predictor or not predictor.is_fitted or not predictor.best_model:
        return info_for_schema

    info_for_schema["model_ready"] = True
    internal_debug_info["preprocessor_fitted"] = predictor.is_fitted
    internal_debug_info["loaded_model_path"] = DEFAULT_MODEL_PATH 
    internal_debug_info["available_models_metrics"] = predictor.performance_metrics
    internal_debug_info["message"] = "Model is loaded."

    if predictor.best_model:
        info_for_schema["model_type"] = predictor.best_model.__class__.__name__
    
    # Update feature_importance_available based on predictor.feature_importance content
    if predictor.feature_importance:
        best_model_type_name = info_for_schema["model_type"]
        specific_importance_found = False
        if best_model_type_name and best_model_type_name in predictor.feature_importance:
            df_best_model_importance = predictor.feature_importance[best_model_type_name]
            if isinstance(df_best_model_importance, pd.DataFrame) and not df_best_model_importance.empty:
                info_for_schema["feature_importance_available"] = True
                specific_importance_found = True
        elif best_model_type_name == "VotingClassifier" and 'Ensemble' in predictor.feature_importance:
            df_ensemble_importance = predictor.feature_importance.get('Ensemble') # Use .get for safety
            if isinstance(df_ensemble_importance, pd.DataFrame) and not df_ensemble_importance.empty:
                info_for_schema["feature_importance_available"] = True
                specific_importance_found = True
        
        if not specific_importance_found: # Check if any other importance data exists
            for df_importance in predictor.feature_importance.values():
                if isinstance(df_importance, pd.DataFrame) and not df_importance.empty:
                    info_for_schema["feature_importance_available"] = True
                    break
    
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
             internal_debug_info["message"] += f" Performance metrics for '{best_model_name_in_metrics}' could not be retrieved or were not a dict."
    
    # print(f"Debug for /model-info. Final info_for_schema: {info_for_schema}")
    # print(f"Debug for /model-info. Internal info: {internal_debug_info}")
    return info_for_schema


def get_feature_importance_data() -> Optional[Dict[str, Any]]:
    predictor = get_predictor()

    if not predictor or not predictor.is_fitted or not predictor.best_model or not predictor.feature_importance:
        return None

    importance_data_to_return: Optional[pd.DataFrame] = None
    model_name_for_importance: Optional[str] = None

    # --- Determine the best key to use for feature importance ---
    best_model_class_name = predictor.best_model.__class__.__name__ # e.g. "VotingClassifier", "RandomForestClassifier"
    
    # 1. Prioritize Permutation Importance for the best model
    potential_perm_key_ensemble = "Ensemble_Permutation"
    potential_perm_key_direct_best_model = f"{best_model_class_name}_Permutation"

    if isinstance(predictor.best_model, VotingClassifier) and potential_perm_key_ensemble in predictor.feature_importance:
        importance_data_to_return = predictor.feature_importance[potential_perm_key_ensemble]
        model_name_for_importance = "Ensemble (Permutation)"
    elif potential_perm_key_direct_best_model in predictor.feature_importance:
        importance_data_to_return = predictor.feature_importance[potential_perm_key_direct_best_model]
        model_name_for_importance = f"{best_model_class_name} (Permutation)"
    
    # 2. Fallback to Direct Importance for the best model (if it's not an ensemble or perm. failed)
    if importance_data_to_return is None:
        potential_direct_key_ensemble = "Ensemble_Direct" # Unlikely for VotingClassifier, but for completeness
        potential_direct_key_best_model = f"{best_model_class_name}_Direct"

        if isinstance(predictor.best_model, VotingClassifier) and potential_direct_key_ensemble in predictor.feature_importance:
            # This would only happen if you calculated and stored direct importance for the ensemble somehow
            importance_data_to_return = predictor.feature_importance[potential_direct_key_ensemble]
            model_name_for_importance = "Ensemble (Direct)"
        elif potential_direct_key_best_model in predictor.feature_importance:
            importance_data_to_return = predictor.feature_importance[potential_direct_key_best_model]
            model_name_for_importance = f"{best_model_class_name} (Direct)"
        elif best_model_class_name in predictor.feature_importance: # Original fallback
             importance_data_to_return = predictor.feature_importance[best_model_class_name]
             model_name_for_importance = f"{best_model_class_name} (Direct/Legacy)"


    # 3. If still nothing, and feature_importance dict is not empty, maybe pick the first available Permutation?
    if importance_data_to_return is None and predictor.feature_importance:
        for key, df in predictor.feature_importance.items():
            if "Permutation" in key and isinstance(df, pd.DataFrame) and not df.empty:
                importance_data_to_return = df
                model_name_for_importance = key.replace("_Permutation", " (Permutation - Fallback)")
                print(f"Warning: Using fallback permutation importance: {model_name_for_importance}")
                break
    
    if importance_data_to_return is None or not isinstance(importance_data_to_return, pd.DataFrame) or importance_data_to_return.empty:
        print("Warning: Final check - No suitable feature importance DataFrame found.")
        return None

    # Ensure required columns exist
    if 'feature' not in importance_data_to_return.columns or 'importance' not in importance_data_to_return.columns:
        print(f"Warning: Feature importance DataFrame for '{model_name_for_importance}' is missing 'feature' or 'importance' columns.")
        return None

    # Convert DataFrame to the list of FeatureImportanceItem objects
    structured_importances: List[FeatureImportanceItem] = []
    for _index, row in importance_data_to_return.iterrows():
        structured_importances.append(
            FeatureImportanceItem(feature=str(row['feature']), importance=float(row['importance']))
        )
    
    return {
        "feature_importances": structured_importances,
        "model_type": model_name_for_importance if model_name_for_importance else "Unknown"
    }