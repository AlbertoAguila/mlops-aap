# Lab A — MLOps Titanic: Design Document

**Dataset:** Titanic (raw CSV from Kaggle)  
**Orchestrator:** Prefect 3.x  
**Experiment tracking / Model Registry:** Weights & Biases  
**Model:** K-Nearest Neighbors (KNN)  
**Pattern A:** Coupled (monolithic) flow  
**Pattern B:** Decoupled flows (data pipeline + training pipeline)

---

## 1. Cleaning & Feature Engineering Operations

All operations are fixed and applied identically in both patterns.

### 1.1 Cleaning

| # | Operation | Column(s) | Detail |
|---|---|---|---|
| C1 | Drop irrelevant columns | `PassengerId`, `Name`, `Ticket`, `Cabin` | No predictive value or excessive nulls |
| C2 | Impute nulls | `Age` | Fill with **median** of the column |
| C3 | Impute nulls | `Embarked` | Fill with **mode** of the column |
| C4 | Impute nulls | `Fare` | Fill with **median** (covers test set edge case) |

### 1.2 Feature Engineering

| # | Operation | Column(s) | Detail |
|---|---|---|---|
| FE1 | Label encoding | `Sex` | `male → 1`, `female → 0` |
| FE2 | One-hot encoding | `Embarked` | Produces `Embarked_C`, `Embarked_Q`, `Embarked_S` (drop_first=True → 2 cols) |
| FE3 | New feature | `FamilySize` | `SibSp + Parch + 1` |
| FE4 | New feature | `IsAlone` | `1 if FamilySize == 1 else 0` |
| FE5 | Standard scaling | `Age`, `Fare`, `Pclass`, `SibSp`, `Parch`, `FamilySize` | `sklearn.preprocessing.StandardScaler` |

### 1.3 Final feature set & target

```
Features: Pclass, Sex, Age, SibSp, Parch, Fare,
          Embarked_Q, Embarked_S, FamilySize, IsAlone
Target:   Survived
```

### 1.4 Fixed hyperparameters

| Parameter | Value |
|---|---|
| `n_neighbors` | 5 |
| `test_size` | 0.2 |
| `random_state` | 42 |
| `metric` | `minkowski` |
| `weights` | `uniform` |

---

## 2. Pattern 1 — Coupled Flow (Monolith)

### 2.1 ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  FLOW: titanic-coupled-pipeline                 │
│                                                                 │
│  ┌──────────────┐   ┌────────────┐   ┌─────────────────────┐  │
│  │ ingest_raw   │──▶│ clean_data │──▶│ feature_engineering │  │
│  │ _data        │   │            │   │                     │  │
│  └──────────────┘   └────────────┘   └──────────┬──────────┘  │
│       │ artifact                                 │             │
│       ▼ raw_data                                 ▼             │
│    [W&B: dataset]                      ┌──────────────────┐   │
│                                        │  split_dataset   │   │
│                                        └────────┬─────────┘   │
│                                                 │             │
│                                        ┌────────▼─────────┐   │
│                                        │   train_model    │   │
│                                        └────────┬─────────┘   │
│                                                 │             │
│                                        ┌────────▼─────────┐   │
│                                        │  evaluate_model  │──▶ wandb.log(metrics)
│                                        └────────┬─────────┘   │
│                                                 │             │
│                                        ┌────────▼─────────┐   │
│                                        │  register_model  │──▶ [W&B: model artifact]
│                                        └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Stage Table

| Task | Description | Inputs | Outputs | W&B Artifacts |
|---|---|---|---|---|
| `ingest_raw_data` | Descarga titanic.csv, lo registra en W&B | CSV path / URL | `pd.DataFrame` | Upload `raw_data` (type: dataset) |
| `clean_data` | C1–C4: drop cols, imputación mediana/moda | `pd.DataFrame` | `pd.DataFrame` | — |
| `feature_engineering` | FE1–FE5: encoding, nuevas features, scaling | `pd.DataFrame` | `pd.DataFrame` | — |
| `split_dataset` | train/test split estratificado | `pd.DataFrame`, config | X_train, X_test, y_train, y_test | — |
| `train_model` | Entrena KNN con hiperparámetros fijos | X_train, y_train, config | `KNeighborsClassifier` | `wandb.config` con hiperparámetros |
| `evaluate_model` | Calcula accuracy, precision, recall, f1 | model, X_test, y_test | `dict` metrics | `wandb.log(metrics)` |
| `register_model` | Serializa modelo con joblib, registra en W&B | model, run, config | — | Upload `titanic-knn` (type: model) |

---

## 3. Pattern 2 — Decoupled Flows

### 3.1 ASCII Diagram

```
╔══════════════════════════════════════════════════════════════╗
║              FLOW A: titanic-data-pipeline                   ║
║                                                              ║
║  ┌──────────────┐   ┌────────────┐   ┌─────────────────┐   ║
║  │ ingest_raw   │──▶│ clean_data │──▶│ feature_engin.  │   ║
║  │ _data        │   │            │   │                 │   ║
║  └──────────────┘   └────────────┘   └────────┬────────┘   ║
║       │ artifact                              │            ║
║       ▼ raw_data                              ▼            ║
║    [W&B: dataset]                   ┌─────────────────┐   ║
║                                     │ persist_dataset │──▶ [W&B: processed_data]
║                                     └─────────────────┘   ║
╚══════════════════════════════════════════════════════════════╝
                              │
                    W&B Artifact: processed_data
                              │
                              ▼
╔══════════════════════════════════════════════════════════════╗
║             FLOW B: titanic-training-pipeline                ║
║                                                              ║
║  ┌──────────────────┐   ┌───────────────┐                  ║
║  │ load_processed   │──▶│ split_dataset │                  ║
║  │ _dataset         │   │               │                  ║
║  └──────────────────┘   └───────┬───────┘                  ║
║                                 │                           ║
║                        ┌────────▼────────┐                 ║
║                        │  train_model    │──▶ wandb.config  ║
║                        └────────┬────────┘                 ║
║                                 │                           ║
║                        ┌────────▼────────┐                 ║
║                        │ evaluate_model  │──▶ wandb.log     ║
║                        └────────┬────────┘                 ║
║                                 │                           ║
║                        ┌────────▼────────┐                 ║
║                        │ register_model  │──▶ [W&B: model]  ║
║                        └─────────────────┘                 ║
╚══════════════════════════════════════════════════════════════╝
```

### 3.2 Stage Table — Flow A (Data Pipeline)

| Task | Description | Inputs | Outputs | W&B Artifacts |
|---|---|---|---|---|
| `ingest_raw_data` | Descarga titanic.csv, registra en W&B | CSV path | `pd.DataFrame` | Upload `raw_data` (type: dataset) |
| `clean_data` | C1–C4: drop, imputación | `pd.DataFrame` | `pd.DataFrame` | — |
| `feature_engineering` | FE1–FE5: encoding, features, scaling | `pd.DataFrame` | `pd.DataFrame` | — |
| `persist_processed_dataset` | Guarda processed.csv, lo sube a W&B | `pd.DataFrame`, run, config | — | Upload `titanic-processed` (type: dataset) |

### 3.3 Stage Table — Flow B (Training Pipeline)

| Task | Description | Inputs | Outputs | W&B Artifacts |
|---|---|---|---|---|
| `load_processed_dataset` | Descarga artefacto `titanic-processed` de W&B | run, config | `pd.DataFrame` | Consume `titanic-processed:latest` |
| `split_dataset` | train/test split | `pd.DataFrame`, config | X_train, X_test, y_train, y_test | — |
| `train_model` | Entrena KNN | X_train, y_train, config | `KNeighborsClassifier` | `wandb.config` |
| `evaluate_model` | accuracy, precision, recall, f1 | model, X_test, y_test | `dict` | `wandb.log(metrics)` |
| `register_model` | Serializa con joblib, registra en W&B | model, run, config | — | Upload `titanic-knn` (type: model) |

---

## 4. Division Criteria (Decoupled Pattern)

**Criterion: Independence of lifecycle and computational reuse**

| Reason | Explanation |
|---|---|
| **Reusability** | Flow A produces `titanic-processed` once; Flow B can re-run N times (different hyperparams, different models) without re-processing data |
| **Independent triggers** | Data can change (new data ingestion) without affecting the training pipeline; training can be triggered on demand |
| **Separation of responsibilities** | Data engineering team owns Flow A; ML team owns Flow B |
| **W&B as contract** | The processed dataset artifact acts as a versioned interface between the two flows, guaranteeing reproducibility |

**Division boundary:** After `feature_engineering` and before `split_dataset`. The processed+encoded+scaled dataset is the artifact that crosses the boundary.

---

## 5. W&B Artifact Registry Summary

| Artifact name | Type | Produced by | Consumed by |
|---|---|---|---|
| `titanic-raw` | dataset | `ingest_raw_data` (both patterns) | `clean_data` |
| `titanic-processed` | dataset | `persist_processed_dataset` (Flow A only) | `load_processed_dataset` (Flow B only) |
| `titanic-knn` | model | `register_model` (both patterns) | — (Model Registry endpoint) |

---

## 6. Project Structure

```
mlops-aap/
├── docs/
│   └── design_document.md         ← this file
├── flows/
│   ├── coupled_flow/
│   │   ├── flow.py                ← @flow titanic-coupled-pipeline
│   │   └── config.yaml            ← all hyperparameters
│   └── decoupled_flows/
│       ├── flow_a_data.py         ← @flow titanic-data-pipeline
│       ├── flow_b_training.py     ← @flow titanic-training-pipeline
│       └── config.yaml
├── src/
│   ├── data/
│   │   ├── ingest.py              ← ingest_raw_data task logic
│   │   └── clean.py               ← clean_data task logic (C1–C4)
│   ├── features/
│   │   └── engineering.py         ← feature_engineering task logic (FE1–FE5)
│   └── models/
│       ├── train.py               ← train_model + evaluate_model
│       └── register.py            ← register_model
├── docker/
│   ├── docker-compose.yml         ← postgres + redis + prefect-server + prefect-services + prefect-worker
│   └── Dockerfile                 ← image with all dependencies
├── deploy_all.sh                  ← single command to deploy both patterns
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 7. Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Orchestrator | Prefect 3.x | Matches professor's example code exactly |
| Experiment tracking | W&B | Required by lab spec |
| Model serialization | `joblib` | Standard for sklearn, better than pickle for large arrays |
| Config format | YAML (not JSON) | More readable for hyperparameters with comments |
| Dataset source | Kaggle Titanic CSV (bundled locally) | Uploaded to W&B once via `wandb_init.py` equivalent |
| Python version | ≥3.11 | Compatible with Prefect 3.x and all deps |
| Scaler persistence | Scaler fitted on train only, saved as W&B artifact alongside model | Prevents data leakage, enables consistent inference |

---

## 8. W&B Run Configuration per Pattern

### Coupled flow run
```python
wandb.init(
    project=config["project"],          # "titanic-mlops"
    job_type="full-pipeline",
    tags=["coupled", "knn", "titanic"],
    config=config,
    save_code=True,
)
```

### Decoupled Flow A run
```python
wandb.init(
    project=config["project"],
    job_type="data-processing",
    tags=["decoupled", "data", "titanic"],
    config=config,
    save_code=True,
)
```

### Decoupled Flow B run
```python
wandb.init(
    project=config["project"],
    job_type="model-training",
    tags=["decoupled", "training", "knn", "titanic"],
    config=config,
    save_code=True,
)
```
