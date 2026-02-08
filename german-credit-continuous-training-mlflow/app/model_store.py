import json
import joblib
from pathlib import Path

REGISTRY_PATH = Path("artifacts/models/model_registry.json")

def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError("model_registry.json not found. Train v1 first (Notebook 03).")
    return json.loads(REGISTRY_PATH.read_text())

def get_current_version() -> str:
    return load_registry()["current_version"]

def load_current_model():
    reg = load_registry()
    v = reg["current_version"]
    path = reg["models"][v]["path"]
    return v, joblib.load(path)

def update_registry(new_version: str, model_path: str, metrics_path: str):
    reg = load_registry()
    reg["current_version"] = new_version
    reg["models"][new_version] = {"path": model_path, "metrics_path": metrics_path}
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))
