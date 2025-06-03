# student-dropout-prediction-api/app/core_utils.py
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List

# Add the ml directory to Python's path
# This allows importing ml_dropout_prediction_system
# Get the absolute path to the project root (student-dropout-prediction-api)
# Assuming core_utils.py is in student-dropout-prediction-api/app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "ml"
sys.path.insert(0, str(ML_DIR.parent)) # Add project root to path

# Now we can import from ml.ml_dropout_prediction_system
from ml.ml_dropout_prediction_system import StudentDropoutPredictor

# --- Global store for the predictor and its features ---
predictor_instance: Optional[StudentDropoutPredictor] = None
trained_feature_columns: Optional[List[str]] = None

DEFAULT_MODEL_PATH = str(ML_DIR / "models" / "student_dropout_model.pkl")
DEFAULT_DATA_DIR = str(ML_DIR / "data")

def load_predictor_model(model_path: str = DEFAULT_MODEL_PATH):
    global predictor_instance, trained_feature_columns
    
    # Instantiate predictor (it can load its own model)
    # The StudentDropoutPredictor's __init__ doesn't take paths,
    # its load_model method does. We need to ensure paths are relative
    # to project root or absolute.
    predictor_instance = StudentDropoutPredictor()
    
    # Correct model path for StudentDropoutPredictor methods
    # It expects paths like "ml/models/model.pkl" if CWD is project root
    # Or an absolute path.
    
    # Determine actual path to use for loading.
    # If model_path is absolute, use it. If relative, assume relative to project root.
    if not Path(model_path).is_absolute():
        load_path = str(PROJECT_ROOT / model_path)
    else:
        load_path = model_path

    if not Path(load_path).exists():
        print(f"⚠️ Model file not found at {load_path}. API will start without a loaded model.")
        print("   Run the training pipeline or ensure the model file exists.")
        predictor_instance = None # Ensure it's None if load fails
        trained_feature_columns = None
        return

    if predictor_instance.load_model(filepath=load_path):
        print(f"✅ Model loaded successfully from {load_path}")
        # Extract and store the feature columns the preprocessor was trained on
        if predictor_instance.preprocessor and hasattr(predictor_instance.preprocessor, 'transformers_'):
            num_cols = predictor_instance.preprocessor.transformers[0][2] if len(predictor_instance.preprocessor.transformers) > 0 else []
            cat_cols = predictor_instance.preprocessor.transformers[1][2] if len(predictor_instance.preprocessor.transformers) > 1 else []
            trained_feature_columns = list(num_cols) + list(cat_cols)
            print(f"🔧 Expected feature columns ({len(trained_feature_columns)}): {trained_feature_columns[:5]}...") # Print first 5
        else:
            print("⚠️ Could not determine trained feature columns from preprocessor.")
            trained_feature_columns = None # Or load from a config if preprocessor structure is unknown
            # This might happen if the loaded model's preprocessor is not as expected.
            # For robustness, you might want to hardcode or save this list during training.
            # For now, we derive it. The schema StudentDataInput must match these.
    else:
        print(f"❌ Failed to load model from {load_path}. Predictor not available.")
        predictor_instance = None
        trained_feature_columns = None

def get_predictor():
    if predictor_instance is None:
        # Attempt to load if not already loaded (e.g., if startup failed but user retries an endpoint)
        # This is optional; typically, if startup fails, endpoints shouldn't work.
        # load_predictor_model() # Or raise an error
        print("Predictor not loaded. Please ensure the model is trained and loaded.")
    return predictor_instance

def get_trained_columns():
    if trained_feature_columns is None and predictor_instance and predictor_instance.preprocessor:
         # Attempt to derive again if it failed initially but predictor got loaded later
        num_cols = predictor_instance.preprocessor.transformers[0][2] if len(predictor_instance.preprocessor.transformers) > 0 else []
        cat_cols = predictor_instance.preprocessor.transformers[1][2] if len(predictor_instance.preprocessor.transformers) > 1 else []
        return list(num_cols) + list(cat_cols)
    return trained_feature_columns

def create_feature_dataframe(student_data_dict: dict) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame from a student data dictionary,
    ensuring all trained columns are present and in the correct order.
    """
    expected_cols = get_trained_columns()
    if not expected_cols:
        raise ValueError("Trained feature columns not available. Cannot create DataFrame for prediction.")
    
    # Create DataFrame from the single dictionary entry
    # student_df = pd.DataFrame([student_data_dict])
    # Reindex to ensure all expected columns are present, filling missing ones with NaN
    # This handles cases where input dict might not have all keys (e.g. optional fields)
    # student_df = student_df.reindex(columns=expected_cols, fill_value=np.nan)
    
    # A more direct way:
    # Ensure student_data_dict only contains keys relevant to expected_cols or handle extra keys
    filtered_data = {col: student_data_dict.get(col) for col in expected_cols}
    student_df = pd.DataFrame([filtered_data], columns=expected_cols)

    return student_df