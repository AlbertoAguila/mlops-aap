# Lab A — MLOps Titanic Workflows

Prefect 3 + Weights & Biases pipeline for Titanic survival prediction.

Two architectural patterns are implemented:

| Pattern | Flows | Description |
|---|---|---|
| **Coupled** (monolith) | `titanic-coupled-pipeline` | Single Prefect flow: ingest → clean → FE → split → train → evaluate → register |
| **Decoupled** (modular) | `titanic-data-pipeline` + `titanic-training-pipeline` | Flow A produces a processed dataset artifact; Flow B consumes it to train and register the model |

---

## Prerequisites

- Python ≥ 3.11
- Docker Desktop (running)
- [Weights & Biases](https://wandb.ai/) account — [get your API key](https://wandb.ai/authorize)
- Git Bash (to run `deploy_all.sh` on Windows)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/AlbertoAguila/mlops-aap.git
cd mlops-aap
git checkout Lab_A_Workflow
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

| Variable | Description |
|---|---|
| `PREFECT_API_URL` | Prefect server URL. Keep default when running locally with Docker. |
| `WANDB_API_KEY` | Your W&B API key. |
| `WANDB_PROJECT` | W&B project name (default: `titanic-mlops`). |
| `WANDB_ENTITY` | Your W&B username or team name. |

> **Never commit `.env`** — it is already in `.gitignore`.

---

## Running all flows (single command)

```bash
bash deploy_all.sh
```

This single command:
1. Starts Prefect infrastructure via Docker Compose
2. Downloads the Titanic dataset and uploads it to W&B as artifact `titanic-raw`
3. Runs the **coupled** pipeline end-to-end
4. Runs **Flow A** (data pipeline) — produces `titanic-processed` artifact
5. Runs **Flow B** (training pipeline) — consumes `titanic-processed`, trains KNN, registers model

After completion:
- Prefect UI: http://localhost:4200
- W&B dashboard: https://wandb.ai/`<your-entity>`/titanic-mlops

---

## Running flows individually

```bash
# Upload dataset to W&B (one-time setup)
python wandb_init.py

# Coupled flow
python -c "from flows.coupled_flow.flow import coupled_pipeline; coupled_pipeline()"

# Decoupled — Flow A (must run before Flow B)
python -c "from flows.decoupled_flows.flow_a_data import data_pipeline; data_pipeline()"

# Decoupled — Flow B
python -c "from flows.decoupled_flows.flow_b_training import training_pipeline; training_pipeline()"
```

---

## Registering deployments in Prefect UI

```bash
# Starts serving all three deployments (blocking process)
python deploy_flows.py
```

Registered deployments visible in the Prefect UI under **Deployments**:

| Deployment | Flow |
|---|---|
| Titanic - Coupled pipeline | `coupled_pipeline` |
| Titanic - Decoupled data pipeline | `data_pipeline` |
| Titanic - Decoupled training pipeline | `training_pipeline` |

---

## Docker infrastructure

```bash
# Start
docker compose -f docker/docker-compose.yml up -d

# Check health
docker compose -f docker/docker-compose.yml ps

# Stop (keep data)
docker compose -f docker/docker-compose.yml down

# Stop and wipe volumes
docker compose -f docker/docker-compose.yml down -v
```

Services started: `postgres`, `redis`, `prefect-server`, `prefect-services`, `prefect-worker`.

---

## Project structure

```
mlops-aap/
├── docs/
│   └── design_document.md          # Architecture diagrams, stage tables, decisions
├── flows/
│   ├── coupled_flow/
│   │   ├── flow.py                 # @flow titanic-coupled-pipeline
│   │   └── config.yaml             # Hyperparameters and artifact names
│   └── decoupled_flows/
│       ├── flow_a_data.py          # @flow titanic-data-pipeline
│       ├── flow_b_training.py      # @flow titanic-training-pipeline
│       └── config.yaml
├── src/
│   ├── data/
│   │   ├── ingest.py               # Task: download raw artifact from W&B
│   │   └── clean.py                # Task: C1–C4 cleaning operations
│   ├── features/
│   │   └── engineering.py          # Task: FE1–FE4 encoding + new features
│   └── models/
│       ├── train.py                # Tasks: split, train (with FE5 scaling), evaluate
│       └── register.py             # Task: serialize model+scaler, register in W&B
├── docker/
│   ├── docker-compose.yml          # Prefect + Postgres + Redis
│   └── Dockerfile
├── wandb_init.py                   # One-time dataset upload to W&B
├── deploy_flows.py                 # Registers deployments with Prefect serve()
├── deploy_all.sh                   # Single command to run everything
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Cleaning & Feature Engineering operations

| # | Type | Operation |
|---|---|---|
| C1 | Cleaning | Drop `PassengerId`, `Name`, `Ticket`, `Cabin` |
| C2 | Cleaning | Impute `Age` → median |
| C3 | Cleaning | Impute `Embarked` → mode |
| C4 | Cleaning | Impute `Fare` → median |
| FE1 | Encoding | `Sex` → label encode (male=1, female=0) |
| FE2 | Encoding | `Embarked` → one-hot, drop_first (→ `Embarked_Q`, `Embarked_S`) |
| FE3 | New feature | `FamilySize = SibSp + Parch + 1` |
| FE4 | New feature | `IsAlone = 1 if FamilySize == 1 else 0` |
| FE5 | Scaling | `StandardScaler` on `Age`, `Fare`, `Pclass`, `SibSp`, `Parch`, `FamilySize` — fitted on X_train only |

**Model:** KNN (`n_neighbors=5`, `metric=minkowski`, `weights=uniform`)  
**Split:** `test_size=0.2`, `random_state=42`, stratified by `Survived`
