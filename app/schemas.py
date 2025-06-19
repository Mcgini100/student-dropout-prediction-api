# student-dropout-prediction-api/app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class StudentDataInput(BaseModel):
    Gender: str
    Age_at_enrollment: int
    Marital_status: str
    Application_mode: str
    Course_type: str
    Academic_program: str
    Previous_qualification: str
    Previous_qualification_grade: float
    Nationality: str
    Mothers_qualification: str
    Fathers_qualification: str
    Mothers_occupation: str
    Fathers_occupation: str
    Displaced: str
    Educational_special_needs: str
    Debtor: str
    Tuition_fees_up_to_date: str
    Scholarship_holder: str
    Unemployment_rate: float
    Inflation_rate: float
    GDP_growth: float
    Sem1_attendance_rate: Optional[float] = None
    Sem1_GPA: Optional[float] = None
    Sem1_curricular_units_credited: Optional[int] = None
    Sem1_curricular_units_enrolled: Optional[int] = None
    Sem1_curricular_units_evaluations: Optional[int] = None
    Sem1_curricular_units_approved: Optional[int] = None
    Sem1_grade: Optional[float] = None
    Sem1_curricular_units_without_evaluations: Optional[int] = None
    Sem2_attendance_rate: Optional[float] = None
    Sem2_GPA: Optional[float] = None
    Sem2_curricular_units_credited: Optional[int] = None
    Sem2_curricular_units_enrolled: Optional[int] = None
    Sem2_curricular_units_evaluations: Optional[int] = None
    Sem2_curricular_units_approved: Optional[int] = None
    Sem2_grade: Optional[float] = None
    Sem2_curricular_units_without_evaluations: Optional[int] = None

    # Allows easy creation with example data
    class Config:
        json_schema_extra = {
            "example": {
                'Gender': 'Female', 'Age_at_enrollment': 20, 'Marital_status': 'Single',
                'Application_mode': 'Online', 'Course_type': 'Daytime', 'Academic_program': 'Business',
                'Previous_qualification': 'High School', 'Previous_qualification_grade': 85.0,
                'Nationality': 'Domestic', 'Mothers_qualification': 'Higher Education', 
                'Fathers_qualification': 'Higher Education',
                'Mothers_occupation': 'Professional', 'Fathers_occupation': 'Professional',
                'Displaced': 'No', 'Educational_special_needs': 'No', 'Debtor': 'No',
                'Tuition_fees_up_to_date': 'Up to date', 'Scholarship_holder': 'Yes',
                'Unemployment_rate': 5.0, 'Inflation_rate': 2.0, 'GDP_growth': 2.5,
                'Sem1_attendance_rate': 90.0, 'Sem1_GPA': 3.5, 'Sem1_curricular_units_credited': 5,
                'Sem1_curricular_units_enrolled': 5, 'Sem1_curricular_units_evaluations': 5,
                'Sem1_curricular_units_approved': 5, 'Sem1_grade': 15.0,
                'Sem1_curricular_units_without_evaluations': 0,
                # Sem2 fields can be null or omitted for new students before Sem2
                'Sem2_GPA': None 
            }
        }
class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float

class FeatureImportanceResponse(BaseModel):
    # Option 1: Two separate lists (as you used in frontend example)
    # features: List[str]
    # importances: List[float]
    # Option 2: List of objects (more structured, and often preferred)
    feature_importances: List[FeatureImportanceItem]
    model_type: Optional[str] = None # To know which model's importance this is
    
class PredictionResponse(BaseModel):
    predicted_outcome: str
    probabilities: Dict[str, float]
    dropout_probability: float
    risk_level: str
    recommendations: List[str]

class StudentPredictionResult(PredictionResponse):
    student_id: Optional[str] = None # For batch results
    error: Optional[str] = None # If prediction failed for this student

class BatchPredictionResponse(BaseModel):
    results: List[StudentPredictionResult]

class TrainResponse(BaseModel):
    status: str
    message: str
    model_path: Optional[str] = None
    insights_generated: Optional[bool] = None

class ModelInfoResponse(BaseModel):
    model_type: Optional[str] = None
    training_date: Optional[str] = None # This would require storing metadata
    performance_metrics: Optional[Dict[str, Any]] = None
    feature_importance_available: bool = False
    model_ready: bool = False