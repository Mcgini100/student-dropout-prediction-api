# student-dropout-prediction-api/Dockerfile
# numpy 2.3 / scipy 1.16 in requirements.txt need Python >= 3.11
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first to leverage the Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The trained model is not committed to git. Train it at build time from the
# bundled (synthetic) dataset so the image ships ready to serve. Seeds are
# fixed, so every build produces the same model (about 1 minute).
RUN python ml/ml_dropout_prediction_system.py \
    && test -f ml/models/student_dropout_model.pkl

EXPOSE 8000

# Hosting platforms such as Render and Cloud Run inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
