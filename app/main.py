# student-dropout-prediction-api/app/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # Ensure this is imported
from typing import List, Dict, Any, Optional # Ensure Optional is imported
import pandas as pd
import io

# Import Pydantic models from schemas.py
from app.schemas import (
    StudentDataInput,
    PredictionResponse,
    BatchPredictionResponse,
    TrainResponse,
    ModelInfoResponse,
    FeatureImportanceItem,  # New schema for feature importance item
    FeatureImportanceResponse # New schema for feature importance response
)
from app.services import (
    predict_single_student,
    predict_batch_students,
    train_model_pipeline,
    get_model_information,
    get_feature_importance_data # New service function
)
from app.core_utils import load_predictor_model, get_predictor, DEFAULT_MODEL_PATH

app = FastAPI(
    title="Student Dropout Prediction API",
    description="API for predicting student dropout risk and academic performance.",
    version="1.0.0"
)

# CORS Configuration
# Adjust origins as needed for your frontend development and production environments
origins = [
    "http://localhost",         # Common base for many local dev servers
    "http://localhost:3000",    # Default for Create React App
    "http://localhost:5173",    # Default for Vite (React, Vue, Svelte)
    "http://localhost:8080",    # Common for Vue CLI
    # Add your production frontend origin here when deploying
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.on_event("startup")
async def startup_event():
    print("🚀 API starting up. Loading ML model...")
    load_predictor_model(model_path=DEFAULT_MODEL_PATH)
    predictor = get_predictor()
    if predictor and predictor.is_fitted:
        print("✅ ML Model loaded and ready.")
    else:
        print("⚠️ ML Model not loaded or not ready. Training may be required via /train endpoint.")


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Student Dropout Prediction API. See /docs for details."}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(student_data: StudentDataInput):
    """
    Predict dropout risk for a single student.
    """
    predictor = get_predictor()
    if not predictor or not predictor.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded or ready. Train or load a model first.")
    try:
        return predict_single_student(student_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(students_data: List[StudentDataInput]):
    """
    Predict dropout risk for a batch of students (JSON input).
    """
    predictor = get_predictor()
    if not predictor or not predictor.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded or ready. Train or load a model first.")
    try:
        results = predict_batch_students(students_data)
        return BatchPredictionResponse(results=results)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during batch prediction: {str(e)}")

@app.post("/batch-predict-csv", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict_csv(file: UploadFile = File(...)):
    """
    Predict dropout risk for students from an uploaded CSV file.
    The CSV must have headers matching the `StudentDataInput` schema.
    """
    predictor = get_predictor()
    if not predictor or not predictor.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded or ready. Train or load a model first.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")

    students_input_list: List[StudentDataInput] = []
    student_ids = df['Student_ID'].tolist() if 'Student_ID' in df.columns else [f"csv_student_{i+1}" for i in range(len(df))]

    if 'Student_ID' in df.columns:
        df_features = df.drop(columns=['Student_ID'])
    else:
        df_features = df
        
    for index, row in df_features.iterrows():
        try:
            data_dict = row.where(pd.notnull(row), None).to_dict()
            students_input_list.append(StudentDataInput(**data_dict))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid data in CSV row {index + 1}: {str(e)}. Ensure columns match schema.")
    
    try:
        prediction_results = predict_batch_students(students_input_list)
        for i, res in enumerate(prediction_results):
            res.student_id = student_ids[i]
        return BatchPredictionResponse(results=prediction_results)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during CSV batch prediction: {str(e)}")


@app.post("/train", response_model=TrainResponse, tags=["Management"])
async def train_model(background_tasks: BackgroundTasks):
    """
    Trigger the ML model training pipeline.
    This is a long-running task. The API currently handles it synchronously.
    For true background execution with status polling, Celery or similar would be better.
    """
    try:
        result = train_model_pipeline()
        return TrainResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Management"])
async def model_info():
    """
    Get information about the currently loaded model.
    """
    info = get_model_information()
    # The Pydantic model ModelInfoResponse will automatically select the fields it needs
    # from the 'info' dictionary returned by get_model_information.
    return ModelInfoResponse(**info)

# --- NEW ENDPOINT for Feature Importance ---
@app.get("/feature-importance", response_model=FeatureImportanceResponse, tags=["Insights"])
async def get_global_feature_importance():
    """
    Get global feature importance scores for the best performing model.
    """
    importance_data_dict = get_feature_importance_data() # Call the new service function
    
    if not importance_data_dict or not importance_data_dict.get("feature_importances"):
        # Check if the model is simply not ready vs. importance specifically not found
        predictor = get_predictor()
        if not predictor or not predictor.is_fitted or not predictor.best_model:
             raise HTTPException(status_code=503, detail="Model not loaded or ready. Train a model first.")
        else:
             raise HTTPException(status_code=404, detail="Feature importance data not available for the current model.")
             
    return FeatureImportanceResponse(**importance_data_dict)