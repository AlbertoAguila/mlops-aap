import pandas as pd
from prefect import task
from prefect.logging import get_run_logger


@task(name="Clean Titanic data")
def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger = get_run_logger()
    # C1: Drop columns with no predictive value or excessive nulls
    df = df.drop(columns=config["drop_columns"], errors="ignore")
    # C2: Impute Age with median
    df["Age"] = df["Age"].fillna(df["Age"].median())
    # C3: Impute Embarked with mode
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    # C4: Impute Fare with median (covers test-set edge cases)
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    remaining_nulls = int(df.isnull().sum().sum())
    logger.info(
        f"Cleaned data: {df.shape[0]} rows, {df.shape[1]} cols, "
        f"remaining nulls={remaining_nulls}"
    )
    return df
