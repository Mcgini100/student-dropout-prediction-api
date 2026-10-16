"""API tests using FastAPI's TestClient.

The trained model file is not committed. The session fixture trains it once
(about a minute on a laptop) if ml/models/student_dropout_model.pkl is missing.
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "ml" / "models" / "student_dropout_model.pkl"
SAMPLE_CSV = ROOT / "ml" / "data" / "dashboard_sample_students.csv"
NOT_A_CSV = ROOT / "test" / "not_a_csv.txt"

VALID_STUDENT = {
    "Gender": "Female", "Age_at_enrollment": 20, "Marital_status": "Single",
    "Application_mode": "Online", "Course_type": "Daytime", "Academic_program": "Business",
    "Previous_qualification": "High School", "Previous_qualification_grade": 85.0,
    "Nationality": "Domestic", "Mothers_qualification": "Higher Education",
    "Fathers_qualification": "Higher Education", "Mothers_occupation": "Professional",
    "Fathers_occupation": "Professional", "Displaced": "No", "Educational_special_needs": "No",
    "Debtor": "No", "Tuition_fees_up_to_date": "Up to date", "Scholarship_holder": "Yes",
    "Unemployment_rate": 5.0, "Inflation_rate": 2.0, "GDP_growth": 2.5,
    "Sem1_attendance_rate": 90.0, "Sem1_GPA": 3.5, "Sem1_curricular_units_credited": 5,
    "Sem1_curricular_units_enrolled": 5, "Sem1_curricular_units_evaluations": 5,
    "Sem1_curricular_units_approved": 5, "Sem1_grade": 15.0,
    "Sem1_curricular_units_without_evaluations": 0,
}


@pytest.fixture(scope="session")
def trained_model():
    if not MODEL_PATH.exists():
        cwd = os.getcwd()
        os.chdir(ROOT)  # the training script uses paths relative to the repo root
        try:
            from ml.ml_dropout_prediction_system import main_training_pipeline
            main_training_pipeline()
        finally:
            os.chdir(cwd)
    assert MODEL_PATH.exists(), "training did not produce a model file"
    return MODEL_PATH


@pytest.fixture(scope="session")
def client(trained_model):
    from app.main import app
    with TestClient(app) as c:  # the context manager runs the startup event (model load)
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Student Dropout Prediction API" in r.json()["message"]


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_input(client):
    r = client.post("/predict", json=VALID_STUDENT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_outcome"] in {"Dropout", "Enrolled", "Graduate"}
    assert set(body["probabilities"]) == {"Dropout", "Enrolled", "Graduate"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= body["dropout_probability"] <= 1.0
    assert body["dropout_probability"] == pytest.approx(body["probabilities"]["Dropout"])
    assert body["risk_level"]
    assert isinstance(body["recommendations"], list)


@pytest.mark.parametrize(
    "payload",
    [
        {k: v for k, v in VALID_STUDENT.items() if k != "Gender"},   # required field missing
        {**VALID_STUDENT, "Age_at_enrollment": "twenty"},              # wrong type
        {},                                                            # empty body
    ],
    ids=["missing-field", "wrong-type", "empty"],
)
def test_predict_invalid_input_returns_422(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_batch_predict_json(client):
    r = client.post("/batch-predict", json=[VALID_STUDENT, {**VALID_STUDENT, "Debtor": "Yes"}])
    assert r.status_code == 200, r.text
    assert len(r.json()["results"]) == 2


def test_batch_predict_csv_upload(client):
    n_rows = sum(1 for _ in SAMPLE_CSV.open()) - 1
    with SAMPLE_CSV.open("rb") as fh:
        r = client.post("/batch-predict-csv", files={"file": ("students.csv", fh, "text/csv")})
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) == n_rows
    assert results[0]["student_id"] == "SMPL001"


def test_batch_predict_csv_rejects_non_csv(client):
    with NOT_A_CSV.open("rb") as fh:
        r = client.post("/batch-predict-csv", files={"file": ("notes.txt", fh, "text/plain")})
    assert r.status_code == 400


def test_batch_predict_csv_rejects_bad_rows(client):
    bad = b"Gender,Age_at_enrollment\nFemale,not-a-number\n"
    r = client.post("/batch-predict-csv", files={"file": ("bad.csv", bad, "text/csv")})
    assert r.status_code == 400


def test_model_info_and_feature_importance(client):
    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["model_type"]
    fi = client.get("/feature-importance")
    assert fi.status_code == 200
    assert len(fi.json()["feature_importances"]) > 0


def test_train_requires_admin_token_when_configured(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    r = client.post("/train")
    assert r.status_code == 403


@pytest.mark.parametrize("path,method,kwargs", [
    ("/predict", "post", {"json": VALID_STUDENT}),
    ("/batch-predict", "post", {"json": [VALID_STUDENT]}),
    ("/batch-predict-csv", "post", {"files": {"file": ("s.csv", b"Gender\nFemale\n", "text/csv")}}),
])
def test_returns_503_when_model_not_loaded(client, monkeypatch, path, method, kwargs):
    import app.core_utils as core_utils
    monkeypatch.setattr(core_utils, "predictor_instance", None)
    r = getattr(client, method)(path, **kwargs)
    assert r.status_code == 503
    assert client.get("/health").json()["model_loaded"] is False
