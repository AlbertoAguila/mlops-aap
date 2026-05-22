import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from prefect import task
from prefect.logging import get_run_logger


@task(name="Split dataset into train/test")
def split_dataset(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger = get_run_logger()
    X = df[config["features"]]
    y = df[config["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )
    logger.info(f"Split: {len(X_train)} train rows, {len(X_test)} test rows (stratified)")
    return X_train, X_test, y_train, y_test


@task(name="Train KNN model")
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    run: wandb.sdk.wandb_run.Run,
    config: dict,
) -> tuple[KNeighborsClassifier, StandardScaler]:
    logger = get_run_logger()
    # Scale only the specified numeric features — fitted on train only to avoid leakage
    scale_cols = [c for c in config["scale_features"] if c in X_train.columns]
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])

    model = KNeighborsClassifier(
        n_neighbors=config["n_neighbors"],
        metric=config["metric"],
        weights=config["weights"],
    )
    model.fit(X_train, y_train)

    run.config.update(
        {
            "n_neighbors": config["n_neighbors"],
            "metric": config["metric"],
            "weights": config["weights"],
            "test_size": config["test_size"],
            "random_state": config["random_state"],
            "scale_features": config["scale_features"],
        }
    )
    logger.info(
        f"Trained KNN — n_neighbors={config['n_neighbors']}, "
        f"metric={config['metric']}, weights={config['weights']}"
    )
    return model, scaler


@task(name="Evaluate model")
def evaluate_model(
    model: KNeighborsClassifier,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run: wandb.sdk.wandb_run.Run,
    config: dict,
) -> dict:
    logger = get_run_logger()
    scale_cols = [c for c in config["scale_features"] if c in X_test.columns]
    X_test = X_test.copy()
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }
    run.log(metrics)
    logger.info(f"Metrics: {metrics}")
    return metrics
