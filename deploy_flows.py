"""Registers all three Prefect deployments and starts serving them.
This process stays alive and listens for trigger events from the Prefect UI/API.
Run this after docker-compose is up.
"""

from dotenv import load_dotenv
load_dotenv()

from prefect import serve

from flows.coupled_flow.flow import coupled_pipeline
from flows.decoupled_flows.flow_a_data import data_pipeline
from flows.decoupled_flows.flow_b_training import training_pipeline

if __name__ == "__main__":
    serve(
        coupled_pipeline.to_deployment(name="Titanic - Coupled pipeline"),
        data_pipeline.to_deployment(name="Titanic - Decoupled data pipeline"),
        training_pipeline.to_deployment(name="Titanic - Decoupled training pipeline"),
    )
