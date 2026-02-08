from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
import json
from datetime import datetime

from app.schemas import CreditInput, PredictRequest, PredictResponse, FeedbackItem, OutcomeFeedback
from app.model_store import load_current_model, get_current_version, update_registry, load_registry
from app.ml.mlflow_utils import setup_mlflow
from app.ml.training import retrain_with_future_and_feedback, FEATURES

app = FastAPI(title="Cooperative Credit Risk (XGBoost) + MLflow")
templates = Jinja2Templates(directory="app/templates")

setup_mlflow()

CLEAN_DATASET_PATH = "artifacts/dataset/german_credit_clean.csv"
FUTURE_POOL_PATH = "artifacts/dataset/future_pool.csv"
FEEDBACK_PATH = "artifacts/dataset/feedback_new_data.csv"

# Nuevos archivos para registrar casos con ID (para "presentación" y trazabilidad)
APPLICATIONS_PATH = "artifacts/dataset/applications.csv"
FEEDBACK_LOG_PATH = "artifacts/dataset/feedback_log.csv"

MODELS_DIR = "artifacts/models"
REPORTS_DIR = "artifacts/reports"
FIGURES_DIR = "artifacts/figures"

def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _append_csv(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(path, index=False)
    return int(len(df_all))

def _load_latest_application_by_id(client_id: str):
    if not os.path.exists(APPLICATIONS_PATH):
        return None
    df = pd.read_csv(APPLICATIONS_PATH)
    if "client_id" not in df.columns:
        return None
    df = df[df["client_id"].astype(str) == str(client_id)]
    if df.empty:
        return None
    # Tomar el último (por created_at si existe, sino por índice)
    if "created_at" in df.columns:
        df = df.sort_values("created_at")
    return df.iloc[-1].to_dict()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "ok", "current_model": get_current_version()}

@app.get("/metrics")
def metrics():
    reg = load_registry()
    v = reg["current_version"]
    metrics_path = reg["models"][v]["metrics_path"]
    with open(metrics_path, "r") as f:
        m = json.load(f)
    return {"current_version": v, "metrics": m}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    version, model = load_current_model()

    # Separar features del ID (si viene)
    client_id = payload.client_id.strip() if payload.client_id else None
    data = payload.model_dump(exclude={"client_id"})
    X = pd.DataFrame([data])[FEATURES]

    proba = float(model.predict_proba(X)[:, 1][0])

    # Policy (demo): >=0.5 "más probable que pague"
    decision = "APROBADO" if proba >= 0.5 else "RECHAZADO"

    # Guardar el caso si viene client_id (para luego registrar el resultado real)
    if client_id:
        row = {"client_id": client_id, **data, "model_version": version, "proba_good": proba, "decision": decision, "created_at": _utc_now_iso()}
        _append_csv(APPLICATIONS_PATH, row)

    return PredictResponse(model_version=version, proba_good=proba, decision=decision)

@app.post("/outcome")
def outcome(item: OutcomeFeedback):
    """
    Registra el resultado real (pagó / no pagó) usando solo el ID.
    - Busca el último caso guardado con ese ID en applications.csv
    - Escribe una fila en feedback_new_data.csv (FEATURES + target) para reentrenamiento
    """
    client_id = item.client_id.strip()
    app_row = _load_latest_application_by_id(client_id)
    if app_row is None:
        raise HTTPException(status_code=404, detail=f"No existe un caso guardado con ID '{client_id}'. Primero haga una evaluación y guarde el caso.")

    target = 1 if item.paid else 0

    # Construir fila para reentrenamiento (sin ID)
    row_features = {k: app_row[k] for k in FEATURES if k in app_row}
    row_features["target"] = int(target)

    df_new = pd.DataFrame([row_features])[FEATURES + ["target"]]
    if os.path.exists(FEEDBACK_PATH):
        df_old = pd.read_csv(FEEDBACK_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(FEEDBACK_PATH, index=False)

    # Log con ID para trazabilidad (no entra al modelo)
    log_row = {"client_id": client_id, "paid": bool(item.paid), "target": int(target), "linked_created_at": app_row.get("created_at", ""), "recorded_at": _utc_now_iso()}
    _append_csv(FEEDBACK_LOG_PATH, log_row)

    return {
        "message": "Resultado real guardado.",
        "client_id": client_id,
        "feedback_rows": int(len(df_all)),
        "note": "Con varios resultados reales (>= 5 recomendado) el reentrenamiento mejora más."
    }

@app.post("/feedback")
def feedback(item: FeedbackItem):
    """
    Endpoint compatible con la versión anterior:
    permite enviar x + target directamente (y opcionalmente client_id).
    """
    os.makedirs("artifacts/dataset", exist_ok=True)

    row = item.x.model_dump()
    row["target"] = int(item.target)
    df_new = pd.DataFrame([row])[FEATURES + ["target"]]

    if os.path.exists(FEEDBACK_PATH):
        df_old = pd.read_csv(FEEDBACK_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(FEEDBACK_PATH, index=False)

    # Si viene client_id, lo dejamos registrado en feedback_log.csv (opcional)
    if item.client_id:
        log_row = {"client_id": item.client_id.strip(), "paid": bool(int(item.target) == 1), "target": int(item.target), "linked_created_at": "", "recorded_at": _utc_now_iso()}
        _append_csv(FEEDBACK_LOG_PATH, log_row)

    return {
        "message": "Feedback saved.",
        "feedback_rows": int(len(df_all)),
        "note": "Retraining becomes more meaningful as feedback grows (>= 5 recommended)."
    }

@app.post("/retrain")
def retrain():
    version, model_path, metrics_path, metrics = retrain_with_future_and_feedback(
    fixed_test_path="artifacts/dataset/fixed_test.csv",
    base50_path="artifacts/dataset/initial_base_50.csv",
    future_pool_path="artifacts/dataset/future_pool.csv",
    feedback_path="artifacts/dataset/feedback_new_data.csv",
    models_dir="artifacts/models",
    reports_dir="artifacts/reports",
    figures_dir="artifacts/figures",
    batch_size=10,
    mlruns_uri="file:./notebooks/mlruns",
    experiment_name="CooperativeCreditRisk-XGBoost",
    debug=False
    )


    update_registry(new_version=version, model_path=model_path, metrics_path=metrics_path)

    return {
        "message": "Model retrained and activated.",
        "new_version": version,
        "model_path": model_path,
        "metrics": metrics,
        "mlflow_ui": "http://127.0.0.1:5000"
    }
