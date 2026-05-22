"""
upload_dataset.py
-----------------
Crea el dataset procesado del iris y lo sube a W&B como artefacto.
Debe ejecutarse UNA VEZ antes de lanzar el flujo de entrenamiento.

Uso:
    uv run python upload_dataset.py

Nota: usa InternalApi directamente en lugar de wandb.init() para evitar
el binario wandb-core que cuelga en sistemas Windows con OneDrive/Defender.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import tempfile
import time

import pandas as pd
from sklearn.datasets import load_iris

import wandb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.lib.hashutil import md5_file_b64
import wandb.filesync.step_prepare as _step_prepare

from utils.config import load_config


def _upload_artifact_direct(
    api: InternalApi,
    entity: str,
    project: str,
    artifact: Artifact,
) -> dict:
    """Upload a wandb Artifact via InternalApi without wandb.init() / wandb-core."""
    manifest = artifact._manifest
    manifest_digest = manifest.digest()

    run_name = f"upload-{artifact.name}-{int(time.time())}"
    run, _, _ = api.upsert_run(
        name=run_name, project=project, entity=entity, job_type="Data Processing"
    )
    run_id = run["name"]

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
        return art_result  # Already uploaded (server-side dedup)

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


def main() -> None:
    config = load_config("wandb")

    print("Cargando dataset Iris...")
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
        "Species",
    ]
    df["PetalAreacm2"] = df["PetalLengthCm"] * df["PetalWidthCm"]
    df["SepalAreacm2"] = df["SepalLengthCm"] * df["SepalWidthCm"]

    print(f"Dataset: {len(df)} filas, columnas: {list(df.columns)}")
    print(df.head(3))

    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, config["processed_filename"])
    df.to_csv(csv_path, index=False)
    print(f"\nCSV guardado en: {csv_path}")

    print(f"\nSubiendo a W&B ({config['entity']}/{config['project']})...")
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    api = InternalApi()

    artifact = Artifact(
        name=config["processed_artifact_name"],
        type="dataset",
        description="Iris dataset procesado con features adicionales (PetalArea, SepalArea)",
    )
    artifact.add_file(csv_path)

    _upload_artifact_direct(api, config["entity"], config["project"], artifact)

    print(f"\n Artefacto '{config['processed_artifact_name']}' subido correctamente.")
    print(f"   Ver en: https://wandb.ai/{config['entity']}/{config['project']}/artifacts")


if __name__ == "__main__":
    main()
