import os
import tempfile

import pandas as pd
import wandb
from prefect import task
from prefect.logging import get_run_logger


@task(name="Ingest raw Titanic data from W&B")
def ingest_raw_data(run: wandb.sdk.wandb_run.Run, config: dict) -> pd.DataFrame:
    logger = get_run_logger()
    artifact_ref = f"{config['raw_artifact_name']}:{config['raw_artifact_version']}"
    artifact = run.use_artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.gettempdir())
    df = pd.read_csv(os.path.join(artifact_dir, config["raw_filename"]))
    logger.info(f"Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    return df
