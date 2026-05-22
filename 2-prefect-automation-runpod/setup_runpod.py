"""
setup_runpod.py
---------------
One-shot script to create the RunPod Serverless Template and Endpoint
for the iris-predict service, then save the generated IDs into
config/runpod.json so all Prefect flows pick them up automatically.

Usage
-----
    uv run python setup_runpod.py

Requirements
------------
- RUNPOD_API_KEY must be set in your .env file or environment.
- The predict Docker image must already exist on Docker Hub
  (run the 'Build and push Docker training image' flow first, then
  the 'Serverless Deployment' flow to build the predict image).
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "runpod.json"
DOCKER_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "docker.json"

RUNPOD_BASE = "https://rest.runpod.io/v1"


def _headers() -> dict:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("[ERROR] RUNPOD_API_KEY is not set. Check your .env file.")
        sys.exit(1)
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# RunPod API calls
# ---------------------------------------------------------------------------

def create_template(image_name: str, container_disk_in_gb: int = 5) -> str:
    """Create a RunPod Serverless template and return its ID."""
    url = f"{RUNPOD_BASE}/templates"
    payload = {
        "containerDiskInGb": container_disk_in_gb,
        "imageName": image_name,
        "name": "iris-predict-template",
        "isServerless": True,
        "env": {},
    }
    resp = requests.post(url, json=payload, headers=_headers())
    if not resp.ok:
        print(f"[ERROR] Failed to create template: {resp.status_code} {resp.text}")
        sys.exit(1)
    data = resp.json()
    template_id: str = data.get("id") or data.get("templateId") or data["data"]["id"]
    print(f"[OK] Template created: id={template_id}  image={image_name}")
    return template_id


def create_endpoint(template_id: str) -> str:
    """Create a RunPod Serverless endpoint and return its ID."""
    url = f"{RUNPOD_BASE}/endpoints"
    payload = {
        "name": "iris-predict-endpoint",
        "templateId": template_id,
        "workersMax": 3,
        "workersMin": 0,
        "idleTimeout": 5,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
    }
    resp = requests.post(url, json=payload, headers=_headers())
    if not resp.ok:
        print(f"[ERROR] Failed to create endpoint: {resp.status_code} {resp.text}")
        sys.exit(1)
    data = resp.json()
    endpoint_id: str = data.get("id") or data.get("endpointId") or data["data"]["id"]
    print(f"[OK] Endpoint created: id={endpoint_id}")
    return endpoint_id


def list_existing_templates() -> list[dict]:
    """List all existing templates to allow reuse."""
    resp = requests.get(f"{RUNPOD_BASE}/templates", headers=_headers())
    if not resp.ok:
        return []
    return resp.json() if isinstance(resp.json(), list) else resp.json().get("data", [])


def list_existing_endpoints() -> list[dict]:
    """List all existing endpoints to allow reuse."""
    resp = requests.get(f"{RUNPOD_BASE}/endpoints", headers=_headers())
    if not resp.ok:
        return []
    return resp.json() if isinstance(resp.json(), list) else resp.json().get("data", [])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    runpod_cfg = load_json(CONFIG_PATH)
    docker_cfg = load_json(DOCKER_CONFIG_PATH)
    image_name: str = docker_cfg["predict"]["image_name"]
    container_disk: int = runpod_cfg["deployment"]["container_disk_in_gb"]

    print("=" * 60)
    print("RunPod Serverless Setup")
    print("=" * 60)
    print(f"Predict image : {image_name}")
    print(f"Disk (GB)     : {container_disk}")
    print()

    # --- Check if placeholders still set ---
    current_template = runpod_cfg["deployment"]["template_id"]
    current_endpoint = runpod_cfg["deployment"]["endpoint_id"]

    if "PLACEHOLDER" not in current_template and "PLACEHOLDER" not in current_endpoint:
        print(
            "[INFO] config/runpod.json already has real IDs:\n"
            f"  template_id : {current_template}\n"
            f"  endpoint_id : {current_endpoint}\n"
            "Nothing to do. Delete them from the config and re-run if you want to recreate."
        )
        return

    # --- Show existing resources so the user can reuse them ---
    print("Fetching existing RunPod resources...")
    existing_templates = list_existing_templates()
    existing_endpoints = list_existing_endpoints()

    iris_templates = [t for t in existing_templates if "iris" in t.get("name", "").lower()]
    iris_endpoints = [e for e in existing_endpoints if "iris" in e.get("name", "").lower()]

    template_id: str | None = None
    endpoint_id: str | None = None

    if iris_templates:
        t = iris_templates[0]
        template_id = t.get("id") or t.get("templateId")
        print(f"[REUSE] Found existing template: id={template_id}  name={t.get('name')}")
    else:
        template_id = create_template(image_name, container_disk)

    if iris_endpoints:
        e = iris_endpoints[0]
        endpoint_id = e.get("id") or e.get("endpointId")
        print(f"[REUSE] Found existing endpoint: id={endpoint_id}  name={e.get('name')}")
    else:
        endpoint_id = create_endpoint(template_id)

    # --- Persist IDs to config ---
    runpod_cfg["deployment"]["template_id"] = template_id
    runpod_cfg["deployment"]["endpoint_id"] = endpoint_id
    save_json(CONFIG_PATH, runpod_cfg)

    print()
    print("=" * 60)
    print("config/runpod.json updated successfully:")
    print(f"  template_id : {template_id}")
    print(f"  endpoint_id : {endpoint_id}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Build and push the training image:  run 'Build and push Docker training image' in Prefect UI")
    print("  2. Train the model:                    run 'Launch RunPod training pod' in Prefect UI")
    print("  3. Deploy serverless endpoint:         run 'Serverless Deployment' in Prefect UI")
    print("  4. Check drift:                        run 'Drift Detection and Retraining Trigger' in Prefect UI")


if __name__ == "__main__":
    main()
