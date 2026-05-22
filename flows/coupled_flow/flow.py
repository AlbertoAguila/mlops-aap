from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from pathlib import Path

import wandb
import yaml
from prefect import flow

from src.data.ingest import ingest_raw_data
from src.data.clean import clean_data
from src.features.engineering import feature_engineering
from src.models.train import split_dataset, train_model, evaluate_model
from src.models.register import register_model


def _load_config() -> dict:
    return yaml.safe_load(
        (Path(__file__).parent / "config.yaml").read_text()
    )


@flow(name="titanic-coupled-pipeline")
def coupled_pipeline() -> None:
    config = _load_config()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        project=os.getenv("WANDB_PROJECT", config["project"]),
        entity=os.getenv("WANDB_ENTITY"),
        job_type=config["job_type"],
        tags=config["tags"],
        config=config,
        save_code=True,
        dir=tempfile.gettempdir(),
    ) as run:
        wandb.save(os.path.relpath(__file__))  # save_code unreliable with Prefect
        df = ingest_raw_data(run, config)
        df = clean_data(df, config)
        df = feature_engineering(df, config)
        X_train, X_test, y_train, y_test = split_dataset(df, config)
        model, scaler = train_model(X_train, y_train, run, config)
        evaluate_model(model, scaler, X_test, y_test, run, config)
        register_model(model, scaler, run, config)
