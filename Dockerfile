# student-dropout-prediction-api/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Create directories for ml module structure
RUN mkdir -p /app/ml/data && mkdir -p /app/ml/models

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project content
COPY . .

# Ensure the ML script is executable if needed (not usually necessary for .py)
# RUN chmod +x ml/ml_dropout_prediction_system.py

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# The CWD will be /app, which is the project root.
# Paths like "ml/models/student_dropout_model.pkl" in StudentDropoutPredictor
# should work correctly from this CWD.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]