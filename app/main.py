# student-dropout-prediction-api/app/main.py
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import pandas as pd
import io

# Import Pydantic models from schemas.py
from app.schemas import (
    StudentDataInput,
    PredictionResponse,
    BatchPredictionResponse,
    TrainResponse,
    ModelInfoResponse,
    FeatureImportanceItem,
    FeatureImportanceResponse,
    AtRiskStats # New schema for dashboard stats
)
from app.services import (
    predict_single_student,
    predict_batch_students,
    train_model_pipeline,
    get_model_information,
    get_feature_importance_data,
    get_dashboard_risk_statistics # New service function for dashboard stats
)
from app.core_utils import load_predictor_model, get_predictor, DEFAULT_MODEL_PATH

app = FastAPI(
    title="Student Dropout Prediction API",
    description="API for predicting student dropout risk and academic performance.",
    version="1.0.0"
)

# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
]
# Extra allowed origins for a deployed frontend, comma-separated, e.g.
# CORS_ORIGINS="https://my-dashboard.onrender.com"
origins += [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.get("/health", tags=["Root"])
async def health():
    """Liveness/readiness probe for the hosting platform."""
    predictor = get_predictor()
    return {"status": "ok", "model_loaded": bool(predictor and predictor.is_fitted)}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(student_data: StudentDataInput):
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
    df_features = df.drop(columns=['Student_ID']) if 'Student_ID' in df.columns else df
    for index, row in df_features.iterrows():
        try:
            data_dict = row.where(pd.notnull(row), None).to_dict()
            students_input_list.append(StudentDataInput(**data_dict))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid data in CSV row {index + 1}: {str(e)}.")
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
async def train_model_endpoint(background_tasks: BackgroundTasks, x_admin_token: Optional[str] = Header(default=None)):
    # Retraining is expensive. When ADMIN_TOKEN is set (as on the public
    # deployment), require it in the X-Admin-Token header.
    admin_token = os.getenv("ADMIN_TOKEN")
    if admin_token and x_admin_token != admin_token:
        raise HTTPException(status_code=403, detail="Retraining requires a valid X-Admin-Token header.")
    try:
        result = train_model_pipeline()
        return TrainResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Management"])
async def model_info_endpoint(): # Renamed
    info = get_model_information()
    return ModelInfoResponse(**info)

@app.get("/feature-importance", response_model=FeatureImportanceResponse, tags=["Insights"])
async def get_global_feature_importance():
    importance_data_dict = get_feature_importance_data()
    if not importance_data_dict or not importance_data_dict.get("feature_importances"):
        predictor = get_predictor()
        if not predictor or not predictor.is_fitted or not predictor.best_model:
             raise HTTPException(status_code=503, detail="Model not loaded or ready. Train a model first.")
        else:
             raise HTTPException(status_code=404, detail="Feature importance data not available for the current model.")
    return FeatureImportanceResponse(**importance_data_dict)

# --- NEW ENDPOINT for Dashboard Statistics ---
@app.get("/dashboard-stats", response_model=Optional[AtRiskStats], tags=["Dashboard"])
async def dashboard_statistics():
    """
    Provides aggregated risk statistics based on a sample of student data.
    Returns null if statistics cannot be generated (e.g., model not ready, no sample data).
    """
    stats_dict = get_dashboard_risk_statistics()
    if not stats_dict:
        # Frontend is designed to handle null response if stats can't be generated
        return None 
        # Alternatively, to be more explicit about failure:
        # raise HTTPException(status_code=404, detail="Could not generate dashboard statistics. Model or sample data may not be ready.")
    return AtRiskStats(**stats_dict)