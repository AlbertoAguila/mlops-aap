import pandas as pd
from prefect import task
from prefect.logging import get_run_logger


@task(name="Feature engineering for Titanic")
def feature_engineering(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger = get_run_logger()
    # FE1: Label-encode Sex (male=1, female=0)
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    # FE2: One-hot encode Embarked — drop_first removes C (alphabetically first)
    #      → produces Embarked_Q and Embarked_S
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    for col in ["Embarked_Q", "Embarked_S"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    # FE3: Total family size on board
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    # FE4: Passenger travelling alone
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    logger.info(
        f"Feature engineering done: {df.shape[1]} cols — {list(df.columns)}"
    )
    return df
