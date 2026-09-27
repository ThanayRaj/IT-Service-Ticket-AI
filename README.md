# IT Service Ticket AI

A portfolio demonstration that predicts a Department and Priority from the text of an IT support ticket. It combines two persisted scikit-learn text-classification pipelines with a Streamlit interface, a small FastAPI intake endpoint, SQLite storage, bulk CSV inference, and a transparent rule-based SLA status indicator.

This is a demonstration project. It is not connected to a real company support system, and its predictions and SLA statuses should not be used for operational decisions.

## Problem and features

Support teams receive tickets that must be routed and prioritized. This project explores whether the ticket's `Body` text can provide useful initial classifications and presents those outputs in an application workflow.

- Predicts Department and Priority from **Body text only** using saved models.
- Accepts single tickets in Streamlit and through `POST /api/tickets`.
- Processes CSV files with a required `Body` column, preserves other columns as metadata, and provides a downloadable result.
- Stores application-created predictions and source labels in SQLite; demo data can be explicitly loaded and selectively cleared.
- Shows ticket history, source counts, and analytics from application records.
- Computes an SLA status from configurable demonstration thresholds. This is a rule-based calculation, not a trained breach predictor.

## Architecture and workflow

```text
Streamlit form / CSV upload / FastAPI request
                    ↓
           TicketApplicationService
                    ↓
           PredictionService
       ┌────────────┴────────────┐
Department pipeline       Priority pipeline
       └────────────┬────────────┘
                    ↓
       Configurable SLA rule engine
                    ↓
             SQLite database
                    ↓
        History and analytics views
```

The prediction service loads `models/department_classifier.joblib` and `models/priority_classifier.joblib` relative to the project, then calls the fitted pipelines as saved. It does not fit models during application use. TF-IDF is part of each fitted pipeline. Bulk columns such as `Department`, `Priority`, and `Tags` are preserved as metadata where present, and are not model inputs.

## Classification models and evaluation

The project uses separate TF-IDF + Linear SVM classifiers for Department and Priority. The experiments compared Logistic Regression, Linear SVM, and Multinomial Naive Bayes; model selection used validation macro F1. Repeated normalized Body values were grouped into a single train, validation, or test partition. TF-IDF was fitted only on each candidate's training data. See the experiment reports for split details, candidate metrics, confusion matrices, and limitations.

| Target | Held-out test accuracy | Held-out test macro F1 |
|---|---:|---:|
| Department | 0.6528 | 0.6555 |
| Priority | 0.6860 | 0.6711 |

These are measured results from one held-out split of the supplied dataset, not a claim of production performance. Priority is imbalanced, and macro F1 is reported alongside accuracy. The models return labels only; the application does not fabricate confidence scores.

## Dataset

The supplied local file is `data/IT Support Ticket Data.csv` (29,651 rows; columns include `Body`, `Department`, `Priority`, and `Tags`, plus an exported index column). It was used for model development. The models use `Body` as their only feature. The file is **not included in this repository** because the project materials do not identify its source or redistribution license. The original local CSV is left unchanged. Reproducing training requires obtaining a properly licensed copy at that path; inference does not need the CSV because the fitted model artifacts are included.

The dataset has repeated ticket bodies, some with conflicting labels, and does not contain historical response times, resolution times, SLA deadlines, or breach outcomes. Those limits apply to model interpretation and rule-based SLA demonstrations.

## Priority and SLA demonstration

For application-created tickets, the SLA engine evaluates ticket age against configurable default thresholds of high: 4 hours, medium: 24 hours, and low: 72 hours. It reports `Within SLA`, `At Risk`, or `Breached`; the at-risk boundary is configurable. Values are demonstration settings, not LTM or any other organization's policy.

**The SLA component is a configurable rule-based demonstration. The project dataset does not contain historical SLA outcomes, so the system does not claim to predict real-world SLA breaches.**

## Streamlit application

The Streamlit entry point is `app/app.py`. Pages cover Overview, Incoming Tickets, Bulk Processing, Ticket History, Analytics, and AI Models. Incoming Tickets is a simulator. Demo tickets are not real customer complaints and load only after a user action. The application does not claim that an external support platform is connected.

The dashboard and analytics use persisted application SQLite records, not the training CSV. Bulk uploads require a `Body` column. The application saves processed nonblank rows with a stable batch identifier and metadata in SQLite so it can restore the latest batch after navigation. Blank Body rows remain in the immediate download with empty predictions and are not stored as tickets. Re-uploading identical CSV content reuses its persisted batch.

## API integration

The optional FastAPI endpoint accepts Body text and calls the same prediction/application service:

```http
POST /api/tickets
Content-Type: application/json

{"body":"The office VPN keeps disconnecting and I cannot access the internal portal."}
```

Run the API locally with `python -m uvicorn src.api:app --reload --port 8000`. Interactive API documentation is available at `/docs`. The endpoint is integration-ready; no customer platform is currently connected.

## SQLite storage and deployment persistence

`src/database.py` creates the `tickets` table automatically. It stores an ID, Body, predicted Department and Priority, UTC creation time, source (`incoming`, `bulk`, `api`, or `demo`), optional idempotency key, optional batch ID/row index, and optional metadata JSON. Existing database files receive additive schema migrations. The local database file is ignored by Git and is not published.

On Streamlit Community Cloud, the local SQLite file is suitable only for a demo. The app's filesystem is not a durable database service: saved ticket records may be lost when the app instance is rebuilt or restarted. Do not store real customer data or rely on it for production persistence. A durable deployment would need an independently managed database, which is outside this project's scope.

## Run locally

Use Python 3.11 or 3.12. Install the pinned runtime packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

The Streamlit app opens at `http://localhost:8501`. To run the API in a separate terminal, use the command above. To install test dependencies, use `python -m pip install -r requirements-dev.txt` and run `python -m pytest`.

The database defaults to `data/application_tickets.sqlite3`. Set `IT_SUPPORT_AI_DATABASE_PATH` to override it; relative override paths resolve from the project root. Model paths are resolved relative to the project and do not depend on the current working directory.

## Deploy on Streamlit Community Cloud

1. Publish this repository to GitHub with the two model files and `.streamlit/config.toml`.
2. In Streamlit Community Cloud, choose **Create app**, select this repository and branch, and set the entry point to `app/app.py`.
3. Select Python 3.12 in Advanced settings (the current models were trained with scikit-learn 1.9.1; the dependency version is pinned in `requirements.txt`). No secrets are required for the demo.
4. Deploy and check the app logs if the initial build fails.

Community Cloud deploys from GitHub and requires repository access. Its free shared resources and filesystem are subject to platform limits; the app may sleep when inactive and its SQLite records are not durable. See [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency configuration](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Repository structure

```text
app/app.py                         Streamlit application
src/prediction.py                  Saved-model prediction service
src/application_service.py         Prediction, persistence, and SLA orchestration
src/database.py                    SQLite repository and additive migrations
src/sla.py                         Configurable SLA demonstration rules
src/api.py                         FastAPI ticket intake endpoint
src/bulk_processing.py             CSV validation and inference
src/train_department.py             Department experiment code
src/train_priority.py              Priority experiment code
models/*.joblib                    Persisted fitted model pipelines
reports/*_model_report.md          Experiment methods and measured results
tests/                             Automated tests
data/                              Local dataset and runtime database (not published)
```

## Technologies

Python, pandas, scikit-learn, TF-IDF, Linear SVM, joblib, Streamlit, FastAPI, Pydantic, and SQLite.
