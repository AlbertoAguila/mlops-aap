import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# Rutas SageMaker Processing (/opt/ml/processing/)
# ─────────────────────────────────────────────────────────────────────────────
INPUT_PATH = "/opt/ml/processing/input/titanic.csv"
TRAIN_PATH = "/opt/ml/processing/output/train"
VAL_PATH   = "/opt/ml/processing/output/validation"
TEST_PATH  = "/opt/ml/processing/output/test"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Carga de datos
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

# 2a. Extraer título del nombre y mapear a categorías numéricas
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_map = {
    "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4,
    "Dr": 5, "Rev": 5, "Col": 5, "Major": 5, "Mlle": 2,
    "Ms": 2, "Lady": 5, "Sir": 5, "Mme": 3,
    "Countess": 5, "Capt": 5, "Jonkheer": 5, "Don": 5, "Dona": 5,
}
df["Title"] = df["Title"].map(title_map).fillna(5).astype(int)

# 2b. Tamaño de familia e indicador de viaje solo
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Imputación de valores nulos
# ─────────────────────────────────────────────────────────────────────────────
df["Age"]      = df["Age"].fillna(df["Age"].median())
df["Fare"]     = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# ─────────────────────────────────────────────────────────────────────────────
# 4. Codificación de variables categóricas
# ─────────────────────────────────────────────────────────────────────────────
df["Sex"]      = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# ─────────────────────────────────────────────────────────────────────────────
# 5. Selección final de columnas
#    IMPORTANTE para XGBoost built-in: 'Survived' debe ser la primera columna
# ─────────────────────────────────────────────────────────────────────────────
features = ["Survived", "Pclass", "Sex", "Age", "Fare",
            "Embarked", "Title", "FamilySize", "IsAlone"]
df = df[features].dropna()

print(f"Shape tras limpieza y feature engineering: {df.shape}")
print(f"Distribución de clases:\n{df['Survived'].value_counts()}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Split train / validation / test (estratificado)
# ─────────────────────────────────────────────────────────────────────────────
train_val, test = train_test_split(
    df, test_size=0.10, random_state=42, stratify=df["Survived"]
)
train, val = train_test_split(
    train_val, test_size=0.15, random_state=42, stratify=train_val["Survived"]
)

print(f"Train: {train.shape} | Validation: {val.shape} | Test: {test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Escritura de salidas
#    Sin cabecera (header=False) y sin índice (index=False) — requisito XGBoost
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(VAL_PATH,   exist_ok=True)
os.makedirs(TEST_PATH,  exist_ok=True)

train.to_csv(os.path.join(TRAIN_PATH, "train.csv"),      header=False, index=False)
val.to_csv(  os.path.join(VAL_PATH,   "validation.csv"), header=False, index=False)
test.to_csv( os.path.join(TEST_PATH,  "test.csv"),        header=False, index=False)

print("Processing job completado correctamente.")
