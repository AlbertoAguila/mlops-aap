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

from src.data.ingest import ingest_raw_data
from src.data.clean import clean_data
from src.features.engineering import feature_engineering


def _load_config() -> dict:
    return yaml.safe_load(
        (Path(__file__).parent / "config.yaml").read_text()
    )


@task(name="Persist processed dataset to W&B")
def persist_processed_dataset(
    df: pd.DataFrame,
    run: wandb.sdk.wandb_run.Run,
    config: dict,
) -> None:
    logger = get_run_logger()
    dataset_path = Path(tempfile.gettempdir()) / config["processed_filename"]
    df.to_csv(dataset_path, index=False)
    artifact = wandb.Artifact(
        name=config["processed_artifact_name"],
        type=config["processed_artifact_type"],
        description="Titanic dataset after cleaning and feature engineering (no scaling)",
    )
    artifact.add_file(str(dataset_path))
    run.log_artifact(artifact)
    logger.info(
        f"Persisted processed dataset as '{config['processed_artifact_name']}' "
        f"({df.shape[0]} rows, {df.shape[1]} cols)"
    )


@flow(name="titanic-data-pipeline")
def data_pipeline() -> None:
    config = _load_config()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        project=os.getenv("WANDB_PROJECT", config["project"]),
        entity=os.getenv("WANDB_ENTITY"),
        job_type=config["process_job_type"],
        tags=config["tags_data"],
        config=config,
        save_code=True,
        dir=tempfile.gettempdir(),
    ) as run:
        wandb.save(os.path.relpath(__file__))
        df = ingest_raw_data(run, config)
        df = clean_data(df, config)
        df = feature_engineering(df, config)
        persist_processed_dataset(df, run, config)
