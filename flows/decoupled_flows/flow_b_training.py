from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from pathlib import Path

import pandas as pd
import wandb
import yaml
from prefect import flow, task
from prefect.logging import get_run_logger

from src.models.train import split_dataset, train_model, evaluate_model
from src.models.register import register_model


def _load_config() -> dict:
    return yaml.safe_load(
        (Path(__file__).parent / "config.yaml").read_text()
    )


@task(name="Load processed Titanic dataset from W&B")
def load_processed_dataset(
    run: wandb.sdk.wandb_run.Run,
    config: dict,
) -> pd.DataFrame:
    logger = get_run_logger()
    artifact_ref = (
        f"{config['processed_artifact_name']}:{config['processed_artifact_version']}"
    )
    artifact = run.use_artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.gettempdir())
    df = pd.read_csv(os.path.join(artifact_dir, config["processed_filename"]))
    logger.info(
        f"Loaded processed artifact '{artifact_ref}': "
        f"{df.shape[0]} rows, {df.shape[1]} cols"
    )
    return df


@flow(name="titanic-training-pipeline")
def training_pipeline() -> None:
    config = _load_config()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        project=os.getenv("WANDB_PROJECT", config["project"]),
        entity=os.getenv("WANDB_ENTITY") or config.get("entity"),
        job_type=config["train_job_type"],
        tags=config["tags_training"],
        config=config,
        save_code=True,
        dir=tempfile.gettempdir(),
    ) as run:
        wandb.save(os.path.relpath(__file__))
        df = load_processed_dataset(run, config)
        X_train, X_test, y_train, y_test = split_dataset(df, config)
        model, scaler = train_model(X_train, y_train, run, config)
        evaluate_model(model, scaler, X_test, y_test, run, config)
        register_model(model, scaler, run, config)
