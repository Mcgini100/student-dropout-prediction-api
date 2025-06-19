# student-dropout-prediction-api/app/core_utils.py
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List # Ensure these are imported

# Add the ml directory to Python's path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "ml"
sys.path.insert(0, str(ML_DIR.parent))

from ml.ml_dropout_prediction_system import StudentDropoutPredictor

# Global store for the predictor and its features
predictor_instance: Optional[StudentDropoutPredictor] = None
trained_feature_columns: Optional[List[str]] = None # This will be set from predictor's attributes

DEFAULT_MODEL_PATH = str(ML_DIR / "models" / "student_dropout_model.pkl")
DEFAULT_DATA_DIR = str(ML_DIR / "data") # Not directly used here but good to define

def load_predictor_model(model_path: str = DEFAULT_MODEL_PATH):
    global predictor_instance, trained_feature_columns
    
    print(f"Attempting to load model from: {model_path}")
    predictor_instance = StudentDropoutPredictor()
    
    load_path = model_path
    if not Path(model_path).is_absolute():
        load_path = str(PROJECT_ROOT / model_path)

    if not Path(load_path).exists():
        print(f"⚠️ Model file not found at {load_path}. API will start without a loaded model.")
        predictor_instance = None
        trained_feature_columns = None
        return

    if predictor_instance.load_model(filepath=load_path):
        print(f"✅ Model loaded successfully from {load_path}")
        # Extract and store the feature columns the preprocessor was trained on
        # These are the *original* feature names before any transformation (e.g. OHE)
        if predictor_instance._all_trained_original_features:
            trained_feature_columns = list(predictor_instance._all_trained_original_features)
            print(f"🔧 Expected original feature columns ({len(trained_feature_columns)}): {trained_feature_columns[:5]}...")
        else:
            print("⚠️ Could not determine original trained feature columns from loaded model predictor instance (_all_trained_original_features).")
            # Fallback or error if this critical info is missing. For robustness, this list should ideally be part of the saved model.
            trained_feature_columns = None 
    else:
        print(f"❌ Failed to load model from {load_path}. Predictor not available.")
        predictor_instance = None
        trained_feature_columns = None

def get_predictor() -> Optional[StudentDropoutPredictor]:
    if predictor_instance is None:
        print("Predictor not loaded. Please ensure the model is trained and loaded.")
    return predictor_instance

def get_trained_original_columns() -> Optional[List[str]]:
    """Returns the list of original feature names the model was trained on."""
    global trained_feature_columns # Explicitly declare usage of global
    if trained_feature_columns is None and predictor_instance and predictor_instance._all_trained_original_features:
         # Attempt to derive again if it failed initially but predictor got loaded later
        trained_feature_columns = list(predictor_instance._all_trained_original_features)
    return trained_feature_columns

def create_feature_dataframe(student_data_dict: dict) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame from a student data dictionary,
    ensuring all original trained columns are present and in the correct order.
    Missing keys in student_data_dict (for expected columns) will result in NaN.
    """
    expected_original_cols = get_trained_original_columns()
    if not expected_original_cols:
        raise ValueError("Original trained feature columns not available from predictor. Cannot create DataFrame for prediction.")
    
    # Create DataFrame ensuring order and presence of all expected_original_cols
    # Values for keys not present in student_data_dict but in expected_original_cols will be NaN
    # Extra keys in student_data_dict not in expected_original_cols will be ignored by pd.DataFrame constructor with 'columns' arg
    
    # Prepare data for DataFrame constructor, ensuring only expected columns are considered from input
    # and missing ones are implicitly handled by pandas (will be NaN or dtype default if not careful)
    # A safer way to ensure all expected columns are present with Nones for missing ones:
    data_for_df = {col: student_data_dict.get(col) for col in expected_original_cols}
    student_df = pd.DataFrame([data_for_df], columns=expected_original_cols)

    return student_df