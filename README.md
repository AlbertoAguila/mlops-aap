# MLOps — Prefect + RunPod + Weights & Biases

This project implements an end-to-end MLOps workflow using [Prefect](https://www.prefect.io/) to orchestrate the training and deployment of a KNN classifier on the Iris dataset.

It integrates:
- **[Weights & Biases](https://wandb.ai/)** — experiment tracking and artifact management (`albertoaguila2003/mlops-aap`)
- **[Docker Hub](https://hub.docker.com/u/albertoaguila)** — container images for training and inference
- **[RunPod](https://runpod.io/)** — cloud training pods and serverless inference endpoints

---

## Project Structure

```
├── config/
│   ├── docker.json        # Docker image names (training & predict)
│   ├── drift.json         # Drift detection settings (KS-test threshold, features)
│   ├── runpod.json        # RunPod pod & serverless deployment settings
│   ├── training.json      # Model hyperparameters, features, train/test split
│   └── wandb.json         # W&B entity, project, artifact names
├── predict/
│   ├── Dockerfile         # Container image for serverless inference
│   ├── download_model.py  # Downloads trained model from W&B artifacts
│   ├── handler.py         # RunPod serverless handler
│   └── requirements.txt   # Predict container dependencies
├── train/
│   ├── Dockerfile         # Container image for the training job (RunPod pod)
│   └── train.py           # Model training script (runs inside the container)
├── utils/
│   ├── config.py          # JSON config loader utility
│   └── runpod/
│       ├── update_endpoint.py  # Update RunPod serverless endpoint via API
│       └── update_template.py  # Update RunPod template image via API
├── deploy_flows.py        # Serves all Prefect flow deployments (single execution)
├── docker-compose.yml     # Prefect server infrastructure (Postgres, Redis, server, worker)
├── f01_build.py           # Flow 1: build/push Docker training image
├── f02_train_pod.py       # Flow 2: launch a RunPod training pod
├── f03_deploy.py          # Flow 3: build/push predict image & update serverless endpoint
├── f04_drift.py           # Flow 4 (Advanced): KS-test drift detection + auto-retrain trigger
├── setup_runpod.py        # One-shot script: create RunPod template & endpoint
└── pyproject.toml         # Python project metadata and dependencies
```

---

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** — must be installed and running
- A **Docker Hub** account (`albertoaguila`) with a Personal Access Token in `DOCKER_PAT`
- A **Weights & Biases** account (`albertoaguila2003`) — API key in `WANDB_API_KEY`
- A **RunPod** account — API key in `RUNPOD_API_KEY`

---

## Environment Setup

1. **Install uv** (if not already installed):

   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create and sync the virtual environment**:

   ```bash
   uv sync
   ```

3. **Create a `.env` file** (copy from `.env.example`) with your secrets:

   ```env
   PREFECT_API_URL=http://127.0.0.1:4200/api
   DOCKER_PAT=your_docker_hub_personal_access_token
   WANDB_API_KEY=your_wandb_api_key
   RUNPOD_API_KEY=your_runpod_api_key
   ```

---

## Starting Prefect

```bash
docker compose up -d
```

This spins up four services:

| Service            | Purpose                          | Port |
|--------------------|----------------------------------|------|
| `postgres`         | Prefect metadata database        | —    |
| `redis`            | Event messaging / cache          | —    |
| `prefect-server`   | API server & UI                  | 4200 |
| `prefect-services` | Background scheduler             | —    |
| `prefect-worker`   | Executes flow runs (local-pool)  | —    |

Open the Prefect UI at **http://localhost:4200**.

---

## RunPod Setup (first time only)

Before running the deployment flow, create the Serverless Template and Endpoint:

```bash
uv run python setup_runpod.py
```

This script:
1. Checks for existing `iris-*` templates/endpoints to avoid duplicates.
2. Creates a Serverless Template pointing to your predict Docker image.
3. Creates a Serverless Endpoint using that template.
4. Saves the generated `template_id` and `endpoint_id` to `config/runpod.json` automatically.

---

## Deploying All Flows (single execution)

```bash
uv run python deploy_flows.py
```

This registers **four deployments** with the Prefect server:

| Deployment | File | Description |
|---|---|---|
| **Build and push Docker training image** | `f01_build.py` | Builds and pushes the training container to Docker Hub |
| **Launch RunPod training pod** | `f02_train_pod.py` | Launches a RunPod pod that runs the training container |
| **Serverless Deployment** | `f03_deploy.py` | Downloads model from W&B, builds/pushes predict image, updates RunPod endpoint |
| **Drift Detection and Retraining Trigger** | `f04_drift.py` | KS-test drift analysis; auto-triggers retraining if drift is found |

Trigger any deployment from the Prefect UI or via:

```bash
prefect deployment run "Flow Name/Deployment Name"
```

---

## Flow Details

### Flow 1 — `f01_build.py`: Build & Push Training Image

Authenticates with Docker Hub, builds the training container (`albertoaguila/runpod-training-iris:latest`) for `linux/amd64`, and pushes it.

### Flow 2 — `f02_train_pod.py`: Launch RunPod Training Pod

Calls the RunPod REST API to create a Pod with the training image. The container:
- Downloads the processed Iris dataset artifact from W&B.
- Trains a KNN classifier.
- Logs metrics (accuracy, precision, recall, F1) to W&B.
- Registers the trained model as a W&B artifact (`iris_knn_model_prefect`).
- Stops the pod automatically after a 60-second grace period.

### Flow 3 — `f03_deploy.py`: Serverless Deployment

1. Downloads the latest trained model from W&B → `predict/model/model.pkl`.
2. Builds and pushes the predict container (`albertoaguila/runpod-predict-iris:latest`).
3. Updates the RunPod Serverless Template to point to the new image.
4. Updates the RunPod Serverless Endpoint to use the updated template.

### Flow 4 — `f04_drift.py`: Drift Detection & Retraining Trigger *(Advanced)*

Detects whether the production data has drifted from the reference baseline:

1. Downloads the **reference dataset** (v0) from W&B.
2. Downloads the **current dataset** (latest) from W&B.
3. Runs the **Kolmogorov-Smirnov two-sample test** on each feature.
4. Logs all KS statistics and p-values to **W&B** as metrics + a JSON artifact.
5. If any feature's p-value falls below `drift_threshold` (default `0.05`), it **automatically triggers** the *Launch RunPod training pod* deployment via the Prefect API.

**Testing drift manually** — to force drift detection, modify the `processed_data` artifact in W&B (or upload a modified CSV), then run this flow. It will detect the distribution shift and kick off retraining.

Configure drift behaviour in `config/drift.json`:

```json
{
  "reference_artifact_version": "v0",
  "drift_threshold": 0.05,
  "features": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "PetalAreacm2", "SepalAreacm2"]
}
```

**Scheduling** — uncomment the `cron` line in `deploy_flows.py` to run drift checks daily at 08:00:
```python
drift_detection_pipeline.to_deployment(
    name="Drift Detection and Retraining Trigger",
    cron="0 8 * * *",
)
```

---

## Serverless Inference

Once deployed, send prediction requests to your RunPod endpoint:

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint_id>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "samples": [{
        "SepalLengthCm": 5.1,
        "SepalWidthCm": 3.5,
        "PetalLengthCm": 1.4,
        "PetalWidthCm": 0.2,
        "PetalAreacm2": 0.28,
        "SepalAreacm2": 17.85
      }]
    }
  }'
```

Response:
```json
{"prediction": [0]}
```

---

## Weights & Biases Project

Results, metrics, and model artifacts are persisted at:
**https://wandb.ai/albertoaguila2003/mlops-aap**

---

## Stopping Services

```bash
docker compose down        # Stop services, keep volumes
docker compose down -v     # Stop services AND remove database/Redis volumes
```

> **Reminder**: Always stop RunPod Pods and check your Serverless Endpoints after use to avoid unnecessary credit consumption.
