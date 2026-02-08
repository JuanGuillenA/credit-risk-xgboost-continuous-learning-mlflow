from pathlib import Path
import mlflow

def setup_mlflow():
    ROOT = Path(__file__).resolve().parents[2]
    MLRUNS = ROOT / "notebooks" / "mlruns"
    mlflow.set_tracking_uri(f"file:{MLRUNS}")
    mlflow.set_experiment("CooperativeCreditRisk-XGBoost")
