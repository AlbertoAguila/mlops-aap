"""One-time script: downloads the Titanic CSV and uploads it to W&B as the
'titanic-raw' dataset artifact.  Run this before executing any flow."""

from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from pathlib import Path

import pandas as pd
import wandb

TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
PROJECT = os.getenv("WANDB_PROJECT", "titanic-mlops")
ARTIFACT_NAME = "titanic-raw"


def main() -> None:
    print(f"Downloading Titanic dataset from {TITANIC_URL} ...")
    df = pd.read_csv(TITANIC_URL)
    csv_path = Path(tempfile.gettempdir()) / "titanic.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved locally: {csv_path}  ({len(df)} rows)")

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        project=PROJECT,
        entity=os.getenv("WANDB_ENTITY"),
        job_type="upload-dataset",
        dir=tempfile.gettempdir(),
    ) as run:
        artifact = wandb.Artifact(
            name=ARTIFACT_NAME,
            type="dataset",
            description="Titanic passenger dataset (Kaggle format, via datasciencedojo/datasets)",
        )
        artifact.add_file(str(csv_path))
        run.log_artifact(artifact)
        print(f"Uploaded '{ARTIFACT_NAME}' to W&B project '{PROJECT}'.")


if __name__ == "__main__":
    main()
