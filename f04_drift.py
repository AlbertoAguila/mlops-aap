from dotenv import load_dotenv
load_dotenv()

import json
import os
import tempfile
import time

import pandas as pd
import wandb
from prefect import flow, task
from prefect.logging import get_run_logger
from scipy import stats

from wandb.sdk.artifacts.artifact import Artifact as WandbArtifact
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.lib.hashutil import md5_file_b64
import wandb.filesync.step_prepare as _step_prepare

from utils.config import load_config


# ---------------------------------------------------------------------------
# Internal helper – mirrors upload_dataset.py (no wandb.init required)
# ---------------------------------------------------------------------------

def _upload_artifact_direct(
    api: InternalApi,
    entity: str,
    project: str,
    run_id: str,
    artifact: WandbArtifact,
) -> dict:
    """Upload a wandb Artifact via InternalApi without wandb.init() / wandb-core."""
    manifest = artifact._manifest
    manifest_digest = manifest.digest()

    art_result, _ = api.create_artifact(
        artifact.type,
        artifact.name,
        manifest_digest,
        entity_name=entity,
        project_name=project,
        run_name=run_id,
        description=artifact.description,
        is_user_created=False,
        client_id=artifact._client_id,
        sequence_client_id=artifact._sequence_client_id,
        aliases=[{"artifactCollectionName": artifact.name, "alias": "latest"}],
    )
    artifact_id = art_result["id"]

    if art_result.get("state") == "COMMITTED":
        return art_result

    manifest_record_id, _ = api.create_artifact_manifest(
        "wandb_manifest.json", "", artifact_id,
        base_artifact_id=None, entity=entity, project=project, run=run_id,
        include_upload=False, type="FULL",
    )

    prep = _step_prepare.StepPrepare(api, 0.1, 0.01, 1000)
    prep.start()
    for _path, entry in manifest.entries.items():
        if entry.local_path is None:
            continue
        resp = prep.prepare({
            "artifactID": artifact_id,
            "artifactManifestID": manifest_record_id,
            "name": entry.path,
            "md5": entry.digest,
        }).get()
        entry.birth_artifact_id = resp.birth_artifact_id
        if resp.upload_url:
            hdrs = dict(h.split(":", 1) for h in (resp.upload_headers or []))
            with open(entry.local_path, "rb") as fh:
                api.upload_file(resp.upload_url, fh, extra_headers=hdrs)
    prep.shutdown()

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as fp:
        manifest_path = fp.name
        json.dump(manifest.to_manifest_json(), fp, indent=4)

    manifest_b64 = md5_file_b64(manifest_path)
    _, mup = api.create_artifact_manifest(
        "wandb_manifest.json", str(manifest_b64), artifact_id,
        base_artifact_id=None, entity=entity, project=project, run=run_id,
    )
    hdrs = dict(h.split(":", 1) for h in mup.get("uploadHeaders", []))
    with open(manifest_path, "rb") as fh:
        api.upload_file(mup["uploadUrl"], fh, extra_headers=hdrs)
    os.unlink(manifest_path)

    api.commit_artifact(artifact_id)
    return art_result


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="Download reference dataset from W&B")
def download_reference_dataset() -> pd.DataFrame:
    """Download the reference (baseline) version of the processed dataset."""
    logger = get_run_logger()
    wandb_cfg = load_config("wandb")
    drift_cfg = load_config("drift")

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    api = wandb.Api()

    artifact_ref = (
        f"{wandb_cfg['entity']}/{wandb_cfg['project']}/"
        f"{wandb_cfg['processed_artifact_name']}:{drift_cfg['reference_artifact_version']}"
    )
    artifact = api.artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.mkdtemp())
    df = pd.read_csv(os.path.join(artifact_dir, wandb_cfg["processed_filename"]))
    logger.info(
        f"Reference dataset loaded (version {drift_cfg['reference_artifact_version']}): "
        f"{len(df)} rows"
    )
    return df


@task(name="Download current dataset from W&B")
def download_current_dataset() -> pd.DataFrame:
    """Download the latest version of the processed dataset."""
    logger = get_run_logger()
    wandb_cfg = load_config("wandb")

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    api = wandb.Api()

    artifact_ref = (
        f"{wandb_cfg['entity']}/{wandb_cfg['project']}/"
        f"{wandb_cfg['processed_artifact_name']}:latest"
    )
    artifact = api.artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.mkdtemp())
    df = pd.read_csv(os.path.join(artifact_dir, wandb_cfg["processed_filename"]))
    logger.info(f"Current dataset loaded (latest): {len(df)} rows")
    return df


@task(name="Run KS drift tests per feature")
def run_ks_tests(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
) -> dict:
    """
    Apply the two-sample Kolmogorov-Smirnov test for each numerical feature.

    Returns a dict with per-feature statistics and an overall drift flag.
    """
    logger = get_run_logger()
    drift_cfg = load_config("drift")
    features: list[str] = drift_cfg["features"]
    threshold: float = drift_cfg["drift_threshold"]

    results: dict[str, dict] = {}
    drift_detected = False

    for feature in features:
        if feature not in df_ref.columns or feature not in df_cur.columns:
            logger.warning(f"Feature '{feature}' not found in one of the datasets — skipping")
            continue

        stat, p_value = stats.ks_2samp(df_ref[feature].dropna(), df_cur[feature].dropna())
        drifted = bool(p_value < threshold)
        if drifted:
            drift_detected = True

        results[feature] = {
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "drifted": drifted,
        }
        status = "DRIFT" if drifted else "OK"
        logger.info(
            f"[{status}] {feature}: KS={stat:.4f}, p={p_value:.4f} "
            f"(threshold={threshold})"
        )

    results["_summary"] = {
        "drift_detected": drift_detected,
        "features_drifted": sum(v["drifted"] for v in results.values() if isinstance(v, dict) and "drifted" in v),
        "total_features": len(features),
        "threshold": threshold,
    }

    if drift_detected:
        logger.warning(
            f"DRIFT DETECTED in "
            f"{results['_summary']['features_drifted']}/{len(features)} features!"
        )
    else:
        logger.info("No drift detected — data distribution is stable.")

    return results


@task(name="Log drift report to W&B")
def log_drift_report(drift_results: dict) -> bool:
    """
    Log drift metrics and a JSON report artifact to Weights & Biases.
    Uses InternalApi directly (no wandb.init / wandb-core) to avoid the
    ServicePollForTokenError on Windows ARM64 + OneDrive/Defender.
    Returns True if drift was detected.
    """
    logger = get_run_logger()
    wandb_cfg = load_config("wandb")
    drift_cfg = load_config("drift")

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    api = InternalApi()

    # Build summary metrics dict
    summary = drift_results.get("_summary", {})
    summary_metrics: dict = {
        "drift/overall_drift_detected": int(summary.get("drift_detected", False)),
        "drift/features_drifted": summary.get("features_drifted", 0),
        "drift/total_features": summary.get("total_features", 0),
    }
    for feature, stats_dict in drift_results.items():
        if feature == "_summary" or not isinstance(stats_dict, dict):
            continue
        summary_metrics[f"drift/{feature}/ks_statistic"] = stats_dict["ks_statistic"]
        summary_metrics[f"drift/{feature}/p_value"] = stats_dict["p_value"]
        summary_metrics[f"drift/{feature}/drifted"] = int(stats_dict["drifted"])

    # Create a W&B run (no wandb-core needed) and attach summary metrics
    run_name = f"drift-{int(time.time())}"
    run_result, _, _ = api.upsert_run(
        name=run_name,
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        job_type=drift_cfg["drift_job_type"],
        summary_metrics=json.dumps(summary_metrics),
    )
    run_id = run_result["name"]

    # Save full JSON report to a temp file
    tmp_dir = tempfile.mkdtemp()
    report_path = os.path.join(tmp_dir, "drift_report.json")
    with open(report_path, "w") as f:
        json.dump(drift_results, f, indent=2)

    # Upload as a W&B artifact (no wandb.init)
    artifact = WandbArtifact(
        name=drift_cfg["drift_artifact_name"],
        type=drift_cfg["drift_artifact_type"],
        description="KS-test drift report comparing reference vs. current dataset",
    )
    artifact.add_file(report_path)

    _upload_artifact_direct(
        api, wandb_cfg["entity"], wandb_cfg["project"], run_id, artifact
    )

    drift_detected: bool = bool(summary.get("drift_detected", False))
    logger.info(f"Drift report logged to W&B project '{wandb_cfg['project']}'")
    return drift_detected


@task(name="Trigger retraining if drift detected")
def trigger_retraining_if_drift(drift_detected: bool) -> None:
    """
    If drift was detected, trigger the training pipeline via the Prefect API.
    This creates a new flow run for the 'Launch RunPod training pod' deployment.
    """
    logger = get_run_logger()

    if not drift_detected:
        logger.info("No drift — retraining not required.")
        return

    logger.warning("Drift detected! Triggering RunPod training pipeline via Prefect API...")

    try:
        from prefect.client.orchestration import get_client
        import asyncio

        async def _trigger():
            async with get_client() as client:
                deployments = await client.read_deployments()
                target = next(
                    (d for d in deployments if "Launch RunPod training pod" in d.name),
                    None,
                )
                if target is None:
                    logger.error(
                        "Deployment 'Launch RunPod training pod' not found. "
                        "Make sure deploy_flows.py is running."
                    )
                    return
                flow_run = await client.create_flow_run_from_deployment(target.id)
                logger.info(
                    f"Created flow run '{flow_run.name}' (id={flow_run.id}) "
                    f"for deployment '{target.name}'"
                )

        asyncio.run(_trigger())

    except Exception as exc:
        logger.error(f"Could not trigger retraining automatically: {exc}")
        logger.info(
            "You can trigger it manually from the Prefect UI "
            "→ 'Launch RunPod training pod' deployment."
        )


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(name="Drift Detection and Retraining Trigger")
def drift_detection_pipeline() -> None:
    """
    Detects dataset drift using the Kolmogorov-Smirnov test, logs results to
    Weights & Biases, and automatically triggers a new RunPod training run if
    drift is detected in any feature.

    Steps
    -----
    1. Download the reference (baseline) dataset from W&B.
    2. Download the current (latest) dataset from W&B.
    3. Run KS tests for each feature.
    4. Log the full drift report to W&B as a metric + JSON artifact.
    5. If drift detected → trigger 'Launch RunPod training pod' deployment.
    """
    df_ref = download_reference_dataset()
    df_cur = download_current_dataset()
    drift_results = run_ks_tests(df_ref, df_cur)
    drift_detected = log_drift_report(drift_results)
    trigger_retraining_if_drift(drift_detected)


if __name__ == "__main__":
    drift_detection_pipeline()
