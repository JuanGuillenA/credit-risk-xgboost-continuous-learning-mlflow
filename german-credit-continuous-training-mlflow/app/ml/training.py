import os, json, time
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier


# =========================
# Features (7 variables)
# =========================
FEATURES = [
    "duration",
    "credit_amount",
    "age",
    "checking_status",
    "employment",
    "savings_status",
    "purpose",
]

CAT_COLS = ["checking_status", "employment", "savings_status", "purpose"]
NUM_COLS = ["duration", "credit_amount", "age"]


def _log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _ensure_columns(df: pd.DataFrame, required: list, df_name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{df_name}] Faltan columnas requeridas: {missing}. "
            f"Columnas actuales: {list(df.columns)}"
        )


def build_pipeline(capacity_tier: str = "v1") -> Pipeline:
    """
    Construye pipeline: imputación + onehot + XGBoost.
    capacity_tier controla la "fuerza" del modelo (demo):
      - v1: menos árboles
      - v2: más árboles (mejor capacidad)
    """
    if capacity_tier == "v1":
        model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=1,            # logs básicos de xgb
        )
    else:
        model = XGBClassifier(
            n_estimators=450,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=1,            # logs básicos de xgb
        )

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, NUM_COLS),
            ("cat", categorical_tf, CAT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])


def evaluate_and_artifacts(model, X_eval, y_eval, prefix: str, figures_dir: str, reports_dir: str):
    """
    Evalúa y genera artifacts:
    - confusion matrix png
    - roc curve png
    - precision-recall png
    - histogram proba png
    - metrics csv
    - metrics json
    """
    proba = model.predict_proba(X_eval)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_eval, pred)),
        "f1": float(f1_score(y_eval, pred)),
        "precision": float(precision_score(y_eval, pred)),
        "recall": float(recall_score(y_eval, pred)),
        "roc_auc": float(roc_auc_score(y_eval, proba)),
    }

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1) Confusion matrix
    cm = confusion_matrix(y_eval, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No pagó", "Pagó"]).plot(ax=ax, values_format="d")
    ax.set_title(f"Matriz de Confusión - {prefix}")
    cm_path = os.path.join(figures_dir, f"{prefix}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) ROC curve
    fpr, tpr, _ = roc_curve(y_eval, proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(f"Curva ROC - {prefix}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    roc_path = os.path.join(figures_dir, f"{prefix}_roc_curve.png")
    fig.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) Precision-Recall
    pr_prec, pr_rec, _ = precision_recall_curve(y_eval, proba)
    pr_auc = average_precision_score(y_eval, proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(pr_rec, pr_prec, label=f"AP={pr_auc:.3f}")
    ax.set_title(f"Precision-Recall - {prefix}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    pr_path = os.path.join(figures_dir, f"{prefix}_pr_curve.png")
    fig.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4) Histograma de probabilidades
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(proba, bins=20)
    ax.set_title(f"Distribución Probabilidad (Pagó=1) - {prefix}")
    ax.set_xlabel("Probabilidad")
    ax.set_ylabel("Frecuencia")
    hist_path = os.path.join(figures_dir, f"{prefix}_proba_hist.png")
    fig.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 5) metrics CSV
    table_path = os.path.join(reports_dir, f"{prefix}_metrics.csv")
    pd.DataFrame([{"version": prefix, **metrics}]).to_csv(table_path, index=False)

    # 6) metrics JSON
    metrics_json_path = os.path.join(reports_dir, f"metrics_{prefix}.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, cm_path, roc_path, pr_path, hist_path, table_path, metrics_json_path


def retrain_with_future_and_feedback(
    fixed_test_path: str,
    base50_path: str,
    future_pool_path: str,
    feedback_path: str,
    models_dir: str,
    reports_dir: str,
    figures_dir: str,
    batch_size: int = 180,
    mlruns_uri: str = "file:./notebooks/mlruns",
    experiment_name: str = "CooperativeCreditRisk-XGBoost",
    capacity_tier: str = "v2",     # fuerza del modelo (no confundir con version timestamp)
    debug: bool = False,
):
    """
    Opción A SIN LEAK:
    - fixed_test.csv: test fijo guardado (NUNCA cambia)
    - initial_base_50.csv: base inicial (sale SOLO del train_pool)
    - future_pool.csv: sale SOLO del train_pool (no toca el test)
    - feedback_new_data.csv: feedback que agrega la app
    """

    t0 = time.time()

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # MLflow
    mlflow.set_tracking_uri(mlruns_uri)
    mlflow.set_experiment(experiment_name)

    _log("========== RETRAIN START ==========")
    _log(f"fixed_test_path={fixed_test_path}")
    _log(f"base50_path={base50_path}")
    _log(f"future_pool_path={future_pool_path}")
    _log(f"feedback_path={feedback_path}")
    _log(f"batch_size={batch_size}")
    _log(f"capacity_tier={capacity_tier}")
    _log(f"mlruns_uri={mlruns_uri}")
    _log(f"experiment={experiment_name}")

    # --------
    # Load FIXED TEST (nunca cambia)
    # --------
    if not os.path.exists(fixed_test_path):
        raise FileNotFoundError(f"fixed_test no encontrado: {fixed_test_path}")

    fixed_test = pd.read_csv(fixed_test_path)
    _ensure_columns(fixed_test, FEATURES + ["target"], "fixed_test")
    fixed_test = fixed_test[FEATURES + ["target"]].copy()
    fixed_test["target"] = fixed_test["target"].astype(int)

    X_eval = fixed_test[FEATURES].copy()
    y_eval = fixed_test["target"].astype(int).copy()

    # --------
    # Load base50
    # --------
    if not os.path.exists(base50_path):
        raise FileNotFoundError(f"initial_base_50 no encontrado: {base50_path}")

    base50 = pd.read_csv(base50_path)
    _ensure_columns(base50, FEATURES + ["target"], "base50")
    base50 = base50[FEATURES + ["target"]].copy()
    base50["target"] = base50["target"].astype(int)

    # --------
    # Load future_pool (de train_pool, sin leak)
    # --------
    if not os.path.exists(future_pool_path):
        raise FileNotFoundError(f"future_pool no encontrado: {future_pool_path}")

    future_pool = pd.read_csv(future_pool_path)
    _ensure_columns(future_pool, FEATURES + ["target"], "future_pool")
    future_pool = future_pool[FEATURES + ["target"]].copy()
    future_pool["target"] = future_pool["target"].astype(int)

    # --------
    # Select batch
    # --------
    take_n = min(batch_size, len(future_pool))
    if take_n == 0:
        batch = future_pool.copy()
        future_pool_remaining = future_pool.copy()
    else:
        # sample para simular llegada aleatoria (si prefieres FIFO, usa .head)
        batch = future_pool.sample(take_n, random_state=7)
        future_pool_remaining = future_pool.drop(batch.index)

    # persist remaining
    future_pool_remaining.to_csv(future_pool_path, index=False)

    # --------
    # Load feedback
    # --------
    if os.path.exists(feedback_path):
        feedback_df = pd.read_csv(feedback_path)
        _ensure_columns(feedback_df, FEATURES + ["target"], "feedback")
        feedback_df = feedback_df[FEATURES + ["target"]].copy()
        feedback_df["target"] = feedback_df["target"].astype(int)
    else:
        feedback_df = pd.DataFrame(columns=FEATURES + ["target"])

    _log(f"rows(base50)={len(base50)}")
    _log(f"rows(future_pool_before)={len(future_pool)})")
    _log(f"rows(future_batch_taken)={len(batch)}")
    _log(f"rows(future_pool_after)={len(future_pool_remaining)}")
    _log(f"rows(feedback)={len(feedback_df)}")
    _log(f"rows(fixed_test)={len(fixed_test)}")

    # --------
    # Train set: batch + feedback
    # --------
    train_plus = pd.concat([base50, batch, feedback_df], ignore_index=True)
    _ensure_columns(train_plus, FEATURES + ["target"], "train_plus")

    X_train = train_plus[FEATURES].copy()
    y_train = train_plus["target"].astype(int).copy()

    _log(f"rows(train_total)={len(train_plus)}")

    # NEW deployment version name (timestamp)
    version = "v_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pipe = build_pipeline(capacity_tier=capacity_tier)

    with mlflow.start_run(run_name=f"retrain_{version}"):

        if debug:
            _log("==== DEBUG COLUMNS ====")
            _log(f"X_train columns={list(X_train.columns)}")
            _log(f"Missing features={[c for c in FEATURES if c not in X_train.columns]}")
            _log(f"Extra columns={[c for c in X_train.columns if c not in FEATURES]}")
            _log(f"Head sample:\n{X_train.head(2)}")
            _log("=======================")

        # Progreso XGBoost: necesitamos transformar primero para pasar eval_set
        fit_t0 = time.time()
        _log("Training model (with XGBoost verbose eval)...")

        pre = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]

        Xtr = pre.fit_transform(X_train)
        Xev = pre.transform(X_eval)

        # fit del modelo con eval_set y verbose => imprime rondas
        model.fit(
            Xtr, y_train,
            eval_set=[(Xev, y_eval)],
            verbose=True
        )

        # el preprocess ya está fitted y el model también, así que el pipeline funciona normal
        _log(f"Training finished in {time.time() - fit_t0:.2f}s")

        # Eval + artifacts con pipeline completo
        metrics, cm_fig, roc_fig, pr_fig, hist_fig, table_csv, metrics_json = evaluate_and_artifacts(
            pipe, X_eval, y_eval, prefix=version, figures_dir=figures_dir, reports_dir=reports_dir
        )

        # Log params
        mlflow.log_param("deployment_version", version)
        mlflow.log_param("capacity_tier", capacity_tier)
        mlflow.log_param("batch_size_used", int(len(batch)))
        mlflow.log_param("future_pool_remaining", int(len(future_pool_remaining)))
        mlflow.log_param("feedback_rows", int(len(feedback_df)))
        mlflow.log_param("train_total_rows", int(len(train_plus)))
        mlflow.log_param("test_rows_fixed", int(len(X_eval)))
        mlflow.log_param("features", ",".join(FEATURES))

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Save model
        model_path = os.path.join(models_dir, f"model_{version}.joblib")
        joblib.dump(pipe, model_path)

        # Log artifacts
        mlflow.log_artifact(model_path, artifact_path="model_joblib")
        mlflow.log_artifact(metrics_json, artifact_path="reports")
        mlflow.log_artifact(table_csv, artifact_path="reports")

        mlflow.log_artifact(cm_fig, artifact_path="figures")
        mlflow.log_artifact(roc_fig, artifact_path="figures")
        mlflow.log_artifact(pr_fig, artifact_path="figures")
        mlflow.log_artifact(hist_fig, artifact_path="figures")

        mlflow.sklearn.log_model(
            pipe,
            artifact_path="sklearn_model",
            registered_model_name=experiment_name
        )


    _log(f"NEW MODEL VERSION = {version}")
    _log(f"Total retrain time = {time.time() - t0:.2f}s")
    _log("========== RETRAIN END ==========")

    return version, model_path, metrics_json, metrics
