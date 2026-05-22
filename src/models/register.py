import tempfile
from pathlib import Path

import joblib
import wandb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from prefect import task
from prefect.logging import get_run_logger


@task(name="Register model in W&B")
def register_model(
    model: KNeighborsClassifier,
    scaler: StandardScaler,
    run: wandb.sdk.wandb_run.Run,
    config: dict,
) -> None:
    logger = get_run_logger()
    tmp = Path(tempfile.gettempdir())
    model_path = tmp / "model.joblib"
    scaler_path = tmp / "scaler.joblib"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    artifact = wandb.Artifact(
        name=config["model_artifact_name"],
        type=config["model_artifact_type"],
        description="KNN classifier + StandardScaler for Titanic survival prediction",
    )
    artifact.add_file(str(model_path))
    artifact.add_file(str(scaler_path))
    run.log_artifact(artifact)
    logger.info(f"Registered model artifact '{config['model_artifact_name']}'")
