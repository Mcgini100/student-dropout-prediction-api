# student-dropout-prediction-api/app/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import pandas as pd
import io
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    StudentDataInput, PredictionResponse, BatchPredictionResponse,
    TrainResponse, ModelInfoResponse
)
from app.services import (
    predict_single_student, predict_batch_students,
    train_model_pipeline, get_model_information, 
)
from app.core_utils import load_predictor_model, get_predictor, DEFAULT_MODEL_PATH

app = FastAPI(
    title="Student Dropout Prediction API",
    description="API for predicting student dropout risk and academic performance.",
    version="1.0.0"
)

origins = [
    "http://localhost:5173",  # Your React frontend origin
    "http://localhost:3000",  # Another common React dev port (optional)
    # You can add other origins here if needed, e.g., your deployed frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    # allow_origins=["*"], # Allows all origins (less secure, use for public APIs or if origins list is too dynamic for dev)
    allow_credentials=True, # Allows cookies to be included in requests (if you use them)
    allow_methods=["*"],    # Allows all standard HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],    # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    print("🚀 API starting up. Loading ML model...")
    load_predictor_model(model_path=DEFAULT_MODEL_PATH) # Load from default path
    predictor = get_predictor()
    if predictor and predictor.is_fitted:
        print("✅ ML Model loaded and ready.")
    else:
        print("⚠️ ML Model not loaded or not ready. Training may be required via /train endpoint.")


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
        # Log e for server-side debugging
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
    except ValueError as e: # From service layer if model not ready for batch
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

    # Convert DataFrame rows to List[StudentDataInput]
    students_input_list: List[StudentDataInput] = []
    # Store student IDs if present
    student_ids = df['Student_ID'].tolist() if 'Student_ID' in df.columns else [f"csv_student_{i+1}" for i in range(len(df))]

    # Drop Student_ID if it was used, as it's not a feature
    if 'Student_ID' in df.columns:
        df_features = df.drop(columns=['Student_ID'])
    else:
        df_features = df
        
    for index, row in df_features.iterrows():
        try:
            # Pydantic will validate and convert types (e.g., NaN from CSV to None for Optional fields)
            data_dict = row.where(pd.notnull(row), None).to_dict() # Convert NaNs to None for Pydantic
            students_input_list.append(StudentDataInput(**data_dict))
        except Exception as e: # PydanticValidationError
            raise HTTPException(status_code=400, detail=f"Invalid data in CSV row {index + 1}: {str(e)}. Ensure columns match schema.")
    
    try:
        prediction_results = predict_batch_students(students_input_list)
        # Map original student IDs back if available
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
    This is a long-running task and will be run in the background.
    The immediate response indicates submission. Check logs/model info for completion.
    """
    # For true background execution and status polling, Celery or similar would be better.
    # Here, BackgroundTasks runs it after response, but client doesn't get live updates.
    
    # Synchronous for now to see output, can be moved to background_tasks.add_task
    # background_tasks.add_task(train_model_pipeline)
    # return TrainResponse(status="submitted", message="Model training pipeline initiated in the background.")
    
    try:
        result = train_model_pipeline() # This is synchronous
        return TrainResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Management"])
async def model_info():
    """
    Get information about the currently loaded model.
    """
    info = get_model_information()
    return ModelInfoResponse(**info)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Student Dropout Prediction API. See /docs for details."}