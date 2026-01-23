# Student Dropout Prediction API

This API provides endpoints for predicting student dropout risk and academic status using machine learning models. It allows for single predictions, batch predictions via JSON or CSV, model training, and retrieving model information.

## Table of Contents

1.  [Running the Backend API](#running-the-backend-api)
    *   [Prerequisites](#prerequisites)
    *   [Setup](#setup)
    *   [Running](#running)
2.  [API Endpoint Documentation](#api-endpoint-documentation)
    *   [Base URL](#base-url)
    *   [Authentication](#authentication)
    *   [Common Responses & Error Codes](#common-responses--error-codes)
    *   [Endpoints](#endpoints)
        *   [GET /](#get-)
        *   [GET /model-info](#get-model-info)
        *   [POST /train](#post-train)
        *   [POST /predict](#post-predict)
        *   [POST /batch-predict](#post-batch-predict)
        *   [POST /batch-predict-csv](#post-batch-predict-csv)
3.  [Data Schemas for Requests & Responses](#data-schemas-for-requests--responses)
    *   [StudentDataInput Schema](#studentdatainput-schema)
    *   [ModelInfoResponse Schema](#modelinforesponse-schema)
    *   [TrainResponse Schema](#trainresponse-schema)
    *   [PredictionResponse Schema](#predictionresponse-schema)
    *   [StudentPredictionResult Schema](#studentpredictionresult-schema)
    *   [BatchPredictionResponse Schema](#batchpredictionresponse-schema)
4.  [CORS Configuration](#cors-configuration)
5.  [Notes for Frontend Development](#notes-for-frontend-development)

---

## 1. Running the Backend API

### Prerequisites

*   Python 3.9+
*   Pip (Python package installer)
*   Access to the project's source code.

### Setup

1.  **Clone the repository (if applicable) or ensure you have the project files.**
2.  **Navigate to the project root directory:**
    ```bash
    cd path/to/student-dropout-prediction-api
    ```
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Dataset:**
    *   Ensure the student dataset (e.g., `student_dropout_dataset.csv`) is placed in the `ml/data/` directory.
    *   If the dataset is not found, the `/train` endpoint will attempt to generate a synthetic dataset and save it as `ml/data/generated_student_dropout_dataset.csv`.
6.  **Pre-trained Model (Optional):**
    *   If a pre-trained model (`student_dropout_model.pkl`) exists in `ml/models/`, it will be loaded on startup.
    *   If no model exists, prediction endpoints will return an error until a model is trained via the `/train` endpoint.

### Running

1.  **Start the FastAPI server using Uvicorn (from the project root directory):**
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload`: Enables auto-reload on code changes (for development).
    *   `--host 0.0.0.0`: Makes the server accessible on your local network.
    *   `--port 8000`: Runs the server on port 8000.
2.  The API will be accessible at `http://localhost:8000`.
3.  Interactive API documentation (Swagger UI) will be available at `http://localhost:8000/docs`.
4.  Alternative OpenAPI documentation (ReDoc) will be at `http://localhost:8000/redoc`.

---

## 2. API Endpoint Documentation

### Base URL

*   Development: `http://localhost:8000`

### Authentication

*   Currently, this API does not implement authentication. All endpoints are publicly accessible once the server is running.

### Common Responses & Error Codes

*   **`200 OK`**: Request was successful.
*   **`201 Created`**: Resource was successfully created (not typically used by these endpoints).
*   **`400 Bad Request`**: The request was malformed (e.g., invalid CSV file, bad parameters). The response body will often contain a `detail` field with more information.
*   **`404 Not Found`**: The requested endpoint does not exist.
*   **`422 Unprocessable Entity`**: The request body was well-formed JSON, but contained invalid data according to the input schema (Pydantic validation error). The response body will contain a `detail` array explaining the validation errors.
*   **`500 Internal Server Error`**: An unexpected error occurred on the server.
*   **`503 Service Unavailable`**: The service is temporarily unavailable (e.g., ML model not loaded yet).

### Endpoints

#### GET `/`

*   **Description:** Welcome endpoint.
*   **Method:** `GET`
*   **Response (200 OK):**
    ```json
    {
      "message": "Welcome to the Student Dropout Prediction API. See /docs for details."
    }
    ```

#### GET `/model-info`

*   **Description:** Retrieves information about the currently loaded machine learning model.
*   **Method:** `GET`
*   **Response (200 OK):** (See [ModelInfoResponse Schema](#modelinforesponse-schema))
    *   If model is loaded:
        ```json
        {
          "model_type": "VotingClassifier", // Example
          "training_date": null,
          "performance_metrics": { "test_accuracy": 0.88, "precision": 0.87, /* ... */ },
          "feature_importance_available": true,
          "model_ready": true
        }
        ```
    *   If model is not loaded:
        ```json
        {
          "model_type": null,
          "training_date": null,
          "performance_metrics": null,
          "feature_importance_available": false,
          "model_ready": false
        }
        ```

#### POST `/train`

*   **Description:** Triggers the machine learning model training pipeline. This can be a long-running process. The API responds immediately indicating submission; actual training happens server-side.
*   **Method:** `POST`
*   **Request Body:** None
*   **Response (200 OK):** (See [TrainResponse Schema](#trainresponse-schema))
    ```json
    {
      "status": "success", // or "failure", "error"
      "message": "Training pipeline completed successfully. Model saved and reloaded.",
      "model_path": "ml/models/student_dropout_model.pkl",
      "insights_generated": false // or true
    }
    ```

#### POST `/predict`

*   **Description:** Predicts dropout risk for a single student.
*   **Method:** `POST`
*   **Request Body:** A JSON object matching the [StudentDataInput Schema](#studentdatainput-schema).
    *   **Content-Type:** `application/json`
*   **Response (200 OK):** (See [PredictionResponse Schema](#predictionresponse-schema))
    ```json
    {
      "predicted_outcome": "Enrolled",
      "probabilities": { "Dropout": 0.15, "Enrolled": 0.70, "Graduate": 0.15 },
      "dropout_probability": 0.15,
      "risk_level": "Low",
      "recommendations": ["Monitor academic progress.", "Encourage engagement in extracurriculars."]
    }
    ```
*   **Response (503 Service Unavailable):** If the model is not loaded/ready.
    ```json
    { "detail": "Model not loaded or ready. Train or load a model first." }
    ```

#### POST `/batch-predict`

*   **Description:** Predicts dropout risk for multiple students provided in a JSON array.
*   **Method:** `POST`
*   **Request Body:** A JSON array where each element is an object matching the [StudentDataInput Schema](#studentdatainput-schema).
    *   **Content-Type:** `application/json`
*   **Response (200 OK):** (See [BatchPredictionResponse Schema](#batchpredictionresponse-schema))
    ```json
    {
      "results": [
        {
          "student_id": "student_input_1",
          "predicted_outcome": "Dropout",
          /* ... other PredictionResponse fields ... */
          "error": null
        },
        {
          "student_id": "student_input_2",
          /* ... */
          "error": "Error during prediction: Some internal issue for this student." // Example error
        }
      ]
    }
    ```

#### POST `/batch-predict-csv`

*   **Description:** Predicts dropout risk for students from an uploaded CSV file.
*   **Method:** `POST`
*   **Request Body:** `multipart/form-data` containing a single file field named `file`.
    *   The CSV file **must** have headers that exactly match the keys of the [StudentDataInput Schema](#studentdatainput-schema).
    *   An optional `Student_ID` column in the CSV will be used for the `student_id` field in the response.
*   **Response (200 OK):** (See [BatchPredictionResponse Schema](#batchpredictionresponse-schema))
    *   Same structure as `/batch-predict`, but `student_id` will be populated from the CSV's `Student_ID` column if present, otherwise auto-generated (e.g., `csv_student_1`).
*   **Response (400 Bad Request):**
    *   If file is not a CSV: `{"detail": "Invalid file type. Please upload a CSV file."}`
    *   If CSV is unparsable: `{"detail": "Error parsing CSV file: <pandas error message>"}`
    *   If CSV data is invalid for a row: `{"detail": "Invalid data in CSV row X: <Pydantic error message>"}`

---

## 3. Data Schemas for Requests & Responses

### StudentDataInput Schema

This schema defines the structure for individual student data used in `/predict` and as elements in the array for `/batch-predict`, and as headers for `/batch-predict-csv`.

**All fields that are numeric but optional (e.g., Semester 2 grades if student dropped out after Sem 1) should be sent as `null` in JSON if not applicable, or be empty cells in CSV.** The API's internal preprocessing handles imputation for `null`/missing numeric values. Categorical fields should always have a valid string value from the expected options.

**Key Fields (Full list is extensive, refer to your dataset's column headers):**

| Field Name                            | Type          | Required? | Description/Example Values                                     |
| ------------------------------------- | ------------- | --------- | -------------------------------------------------------------- |
| `Gender`                              | string        | Yes       | "Male", "Female", "Other"                                      |
| `Age_at_enrollment`                   | integer       | Yes       | e.g., 20                                                       |
| `Marital_status`                      | string        | Yes       | "Single", "Married", "Divorced", ...                           |
| `Application_mode`                    | string        | Yes       | "Online", "In-person", ...                                     |
| `Course_type`                         | string        | Yes       | Name of the academic degree/program e.g. "Biofuel Production Technologies" (Note: In some data this might be 'Course')|
| `Academic_program`                    | string        | Yes       | If distinct from `Course_type`, specify. Otherwise may be redundant. |
| `Previous_qualification`              | string        | Yes       | "Secondary education", "Higher education - bachelor's degree", ... |
| `Previous_qualification_grade`        | number        | Yes       | e.g., 75.5                                                     |
| `Nationality`                         | string        | Yes       | "Portuguese", "Brazilian", ...                                 |
| `Mothers_qualification`               | string        | Yes       | (Same options as Previous_qualification generally)             |
| `Fathers_qualification`               | string        | Yes       | (Same options as Previous_qualification generally)             |
| `Mothers_occupation`                  | string        | Yes       | "Student", "Unskilled Workers", ...                            |
| `Fathers_occupation`                  | string        | Yes       | "Student", "Specialists in Intellectual and Scientific Activities", ... |
| `Displaced`                           | string        | Yes       | "Yes", "No"                                                    |
| `Educational_special_needs`           | string        | Yes       | "Yes", "No"                                                    |
| `Debtor`                              | string        | Yes       | "Yes", "No"                                                    |
| `Tuition_fees_up_to_date`             | string        | Yes       | "Yes" (up to date), "No" (not up to date)                      |
| `Scholarship_holder`                  | string        | Yes       | "Yes", "No"                                                    |
| `Unemployment_rate`                   | number        | Yes       | e.g., 12.1                                                     |
| `Inflation_rate`                      | number        | Yes       | e.g., 1.5                                                      |
| `GDP_growth`                          | number        | Yes       | e.g., 0.7 (Note: in some data this might be absolute 'GDP')  |
| `Sem1_curricular_units_credited`    | integer       | Yes       | e.g., 5                                                        |
| `Sem1_curricular_units_enrolled`    | integer       | Yes       | e.g., 6                                                        |
| `Sem1_curricular_units_evaluations` | integer       | Yes       | e.g., 6                                                        |
| `Sem1_curricular_units_approved`    | integer       | Yes       | e.g., 4                                                        |
| `Sem1_grade`                          | number        | Yes       | Overall grade for Semester 1, e.g., 13.5                       |
| `Sem1_curricular_units_without_evaluations`| integer  | Yes       | e.g., 0                                                        |
| `Sem2_curricular_units_credited`    | integer (opt) | Yes       | `null` if not applicable (e.g., dropout after Sem1)            |
| `Sem2_curricular_units_enrolled`    | integer (opt) | Yes       | `null` if not applicable                                       |
| `Sem2_curricular_units_evaluations` | integer (opt) | Yes       | `null` if not applicable                                       |
| `Sem2_curricular_units_approved`    | integer (opt) | Yes       | `null` if not applicable                                       |
| `Sem2_grade`                          | number (opt)  | Yes       | `null` if not applicable                                       |
| `Sem2_curricular_units_without_evaluations`| integer (opt)| Yes   | `null` if not applicable                                       |
| *... (Potentially other engineered features like `Sem1_GPA`, `Sem1_attendance_rate` if used by the model)* |               |           |                                                                |

**Note for Frontend Developers:** The exact list of categorical options for `select` dropdowns should be obtained from the team that developed the model or by inspecting the unique values in the training dataset for each categorical column. Using incorrect string values will lead to prediction errors.

### ModelInfoResponse Schema

```json
{
  "model_type": "string | null",
  "training_date": "string | null",
  "performance_metrics": "object | null", // e.g., {"test_accuracy": 0.88, "precision": 0.87, ...}
  "feature_importance_available": "boolean",
  "model_ready": "boolean"
}

TrainResponse Schema
{
  "status": "string", // "success", "failure", or "error"
  "message": "string",
  "model_path": "string | null",
  "insights_generated": "boolean | null"
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END
PredictionResponse Schema

This is the structure for a single prediction result.

{
  "predicted_outcome": "string", // e.g., "Dropout", "Enrolled", "Graduate"
  "probabilities": {
    "Dropout": "number",  // Probability of being a dropout
    "Enrolled": "number", // Probability of remaining enrolled
    "Graduate": "number"  // Probability of graduating
    // Keys match the classes the model predicts
  },
  "dropout_probability": "number", // Specific probability for the "Dropout" class
  "risk_level": "string",        // e.g., "Low", "Medium", "High"
  "recommendations": ["string"]  // Array of recommendation strings
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END
StudentPredictionResult Schema

This extends PredictionResponse for batch results.

{
  // Inherits all fields from PredictionResponse
  "predicted_outcome": "string",
  "probabilities": { "...": "number" },
  "dropout_probability": "number",
  "risk_level": "string",
  "recommendations": ["string"],
  // Additional fields for batch context
  "student_id": "string | null", // From CSV or auto-generated for JSON batch
  "error": "string | null"     // Error message if prediction failed for this specific student
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END
BatchPredictionResponse Schema
{
  "results": [
    // Array of StudentPredictionResult objects
  ]
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END
4. CORS Configuration

The backend API is configured with CORS (Cross-Origin Resource Sharing) middleware. By default, it allows requests from:

http://localhost:5173 (Common Vite dev server port)

http://localhost:3000 (Common Create React App dev server port)

If your frontend is running on a different origin during development, this list in the backend's app/main.py will need to be updated, or you can temporarily set allow_origins=["*"] in the backend for easier development (not recommended for production).

5. Notes for Frontend Development

API Base URL: All API calls should be prefixed with http://localhost:8000. Consider making this configurable.

Long-Running Tasks: The /train endpoint initiates a server-side process that can take time. The frontend should provide appropriate user feedback (e.g., loading indicator, status messages) and not block UI interaction. Consider polling /model-info periodically after triggering training to update the UI, or provide a manual refresh button.

Error Handling: Implement robust error handling for API requests. Display meaningful messages to the user based on the HTTP status codes and detail messages from the API.

422 errors from Pydantic will provide detailed validation messages per field.

400 errors for CSV issues will provide specific reasons.

StudentDataInput: The most complex part for the frontend will be creating the form for StudentDataInput. Ensure all fields are represented with appropriate input types. For categorical fields, use dropdowns (<select>) populated with the exact string values the model expects.

Optional Fields: Numeric fields that are optional (like Semester 2 data) can be sent as null in JSON if the user doesn't provide a value. The backend's preprocessing should handle imputation.

CSV Upload: The /batch-predict-csv endpoint expects multipart/form-data. Use a standard file input (<input type="file">) and construct FormData for the request.

User Experience:

Provide clear loading states for all API interactions.

Display prediction results and recommendations in an easy-to-understand format.

For batch results, a table is a good way to present the information.

Guide users on the expected format for CSV files (headers).

