from dotenv import load_dotenv
load_dotenv()

from prefect import serve

from f01_build import build_and_push_pipeline
from f02_train_pod import launch_runpod_pod_pipeline
from f03_deploy import deploy_predict_pipeline
from f04_drift import drift_detection_pipeline

if __name__ == "__main__":
    serve(
        build_and_push_pipeline.to_deployment(
            name="Build and push Docker training image",
        ),
        launch_runpod_pod_pipeline.to_deployment(
            name="Launch RunPod training pod",
        ),
        deploy_predict_pipeline.to_deployment(
            name="Serverless Deployment",
        ),
        drift_detection_pipeline.to_deployment(
            name="Drift Detection and Retraining Trigger",
            # Uncomment to run drift check every day at 08:00:
            # cron="0 8 * * *",
        ),
    )
