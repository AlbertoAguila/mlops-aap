"""
Titanic dataset – drift analysis with EvidentlyAI.

Para cada una de las 12 condiciones experimentales (2 estratificación × 3
ratios × 2 semillas) divide el dataset en train/val/test, usa train como
referencia y genera reports HTML de drift contra val y test. Al final
construye una tabla resumen con la fracción de columnas con drift.

Ejecutar desde la raíz del proyecto `lab A`:

    uv add scikit-learn          # primera vez, si aún no está
    uv run entrega/scripts/run_titanic_drift.py
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent                  # .../lab A/entrega
LAB_ROOT = PROJECT_ROOT.parent              # .../lab A
DATA_PATH = LAB_ROOT / "titanic-dataset.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "resultados"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

for d in (REPORTS_DIR, DOCS_DIR, SPLITS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Esquema de columnas
# ---------------------------------------------------------------------------
# Identificadores y texto libre que no aportan a un análisis de drift estándar
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]
NUMERICAL = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL = ["Survived", "Pclass", "Sex", "Embarked"]
ALL_COLS = NUMERICAL + CATEGORICAL
TARGET = "Survived"

DATA_DEF = DataDefinition(
    numerical_columns=NUMERICAL,
    categorical_columns=CATEGORICAL,
)

# ---------------------------------------------------------------------------
# Condiciones experimentales: 2 × 3 × 2 = 12
# ---------------------------------------------------------------------------
STRATIFY_OPTIONS = [False, True]
RATIO_OPTIONS = {
    # nombre legible: (train, val, test)
    "60-20-20": (0.60, 0.20, 0.20),
    "90-05-05": (0.90, 0.05, 0.05),
    "98-01-01": (0.98, 0.01, 0.01),
}
SEED_OPTIONS = [42, 7]


# ---------------------------------------------------------------------------
# Funciones reutilizables
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    """Carga el Titanic crudo y descarta columnas no usadas para drift."""
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    # Asegurar tipos categóricos consistentes para Evidently
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("object")
    return df


def split_three_way(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    *,
    stratify: bool,
    random_state: int,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide df en (train, val, test) respetando las fracciones dadas.

    Si `stratify` es True, se estratifica sobre la variable objetivo en ambos
    cortes (train vs resto, y val vs test).
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Las fracciones deben sumar 1, recibido {total}")

    strat_col = df[target] if stratify else None
    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=random_state,
        stratify=strat_col,
    )

    # Repartir temp entre val y test manteniendo la proporción relativa
    val_relative = val_frac / (val_frac + test_frac)
    strat_temp = temp_df[target] if stratify else None
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative,
        random_state=random_state,
        stratify=strat_temp,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def to_evidently(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, data_definition=DATA_DEF)


def build_report() -> Report:
    """DataDriftPreset + ValueDrift por columna (para conteo robusto)."""
    metrics = [DataDriftPreset()]
    for col in ALL_COLS:
        metrics.append(ValueDrift(column=col))
    return Report(metrics, include_tests=True)


def _count_drift(
    snap_dict: dict, expected_cols: list[str]
) -> tuple[int, int, dict[str, bool]]:
    """Cuenta columnas con drift de forma robusta a la versión de Evidently.

    Estrategia:
    1. Buscamos en `snap_dict["metrics"]` cualquier entrada cuyo `metric_id`
       contenga "ValueDrift" o "ColumnDrift" (varias versiones).
    2. Para cada una intentamos sacar el nombre de columna del propio
       `metric_id`, y el flag de drift desde `value.drift_detected`,
       `value.share`, o desde un test cuyo `status` sea FAIL.
    """
    import re

    metrics = snap_dict.get("metrics", []) or []
    per_col: dict[str, bool] = {}

    for m in metrics:
        mid = str(m.get("metric_id", ""))
        if not re.search(r"(ValueDrift|ColumnDrift)", mid, re.IGNORECASE):
            continue

        # Extraer la columna
        col_name = None
        for col in expected_cols:
            # match "column=Age", "column='Age'", "column=\"Age\"" o similares
            if re.search(rf"column\s*=\s*['\"]?{re.escape(col)}['\"]?", mid):
                col_name = col
                break
        if col_name is None:
            # Fallback: lo que haya entre "column=" y ")" o ","
            mo = re.search(r"column\s*=\s*['\"]?([^,'\")]+)", mid)
            if mo:
                col_name = mo.group(1).strip()
        if col_name is None:
            continue

        drifted = False

        # 1. Tests fallidos
        for t in m.get("tests", []) or []:
            status = str(t.get("status", "")).upper()
            if status in ("FAIL", "FAILED"):
                drifted = True
                break

        # 2. value.drift_detected
        if not drifted:
            value = m.get("value")
            if isinstance(value, dict):
                if value.get("drift_detected") is True:
                    drifted = True
                elif "drift_score" in value and "stattest_threshold" in value:
                    try:
                        drifted = float(value["drift_score"]) > float(
                            value["stattest_threshold"]
                        )
                    except Exception:
                        pass

        # 3. result.drift_detected (algunas versiones usan "result" en vez de "value")
        if not drifted:
            result = m.get("result")
            if isinstance(result, dict) and result.get("drift_detected") is True:
                drifted = True

        per_col[col_name] = drifted

    n_drifted = sum(1 for v in per_col.values() if v)
    return n_drifted, len(per_col), per_col


_FIRST_DEBUG_DUMP_DONE = False


def run_drift_report(
    reference: Dataset, current: Dataset, html_path: Path, label: str
) -> dict:
    global _FIRST_DEBUG_DUMP_DONE

    report = build_report()
    snap = report.run(current_data=current, reference_data=reference)
    snap.save_html(str(html_path))

    snap_dict = snap.dict()

    # En el primer caso volcamos los metric_ids para depurar formato
    if not _FIRST_DEBUG_DUMP_DONE:
        debug_path = DOCS_DIR / "_debug_first_snapshot.json"
        debug_payload = {
            "metric_ids": [str(m.get("metric_id", "")) for m in snap_dict.get("metrics", [])],
            "first_metric_keys": list(
                (snap_dict.get("metrics", [{}])[0] or {}).keys()
            ),
            "first_metric_value": (snap_dict.get("metrics", [{}])[0] or {}).get("value"),
        }
        debug_path.write_text(json.dumps(debug_payload, indent=2, default=str), encoding="utf-8")
        _FIRST_DEBUG_DUMP_DONE = True
    n_drifted, n_total, per_col = _count_drift(snap_dict, ALL_COLS)

    # Fallback final si seguimos sin encontrar nada: agregamos a ojo desde
    # cualquier campo que se llame "share" / "share_of_drifted_columns".
    if n_total == 0:
        for m in snap_dict.get("metrics", []):
            value = m.get("value") or m.get("result") or {}
            if isinstance(value, dict) and "share" in value and "count" in value:
                count = int(value.get("count") or 0)
                share_val = float(value.get("share") or 0.0)
                total = round(count / share_val) if share_val else len(ALL_COLS)
                n_drifted, n_total = count, total
                break

    share = (n_drifted / n_total) if n_total else 0.0

    print(
        f"  - {label:<10} -> drift en {n_drifted}/{n_total} "
        f"({share:.1%})  [{html_path.name}]"
    )
    return {
        "label": label,
        "html": str(html_path.relative_to(PROJECT_ROOT)),
        "n_drifted": n_drifted,
        "n_total": n_total,
        "share_drift": round(share, 4),
        "per_column_drift": per_col,
    }


# ---------------------------------------------------------------------------
# Loop experimental
# ---------------------------------------------------------------------------
def make_case_id(stratify: bool, ratio_name: str, seed: int) -> str:
    s = "strat" if stratify else "nostrat"
    return f"{s}_{ratio_name}_seed{seed}"


def run_all() -> list[dict]:
    df = load_dataset(DATA_PATH)
    print(f"Dataset Titanic cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"Columnas usadas: {ALL_COLS}\n")

    rows: list[dict] = []

    combos = product(STRATIFY_OPTIONS, RATIO_OPTIONS.items(), SEED_OPTIONS)
    for stratify, (ratio_name, fracs), seed in combos:
        case_id = make_case_id(stratify, ratio_name, seed)
        print(
            f"[CASO] {case_id}  | stratify={stratify}  ratio={ratio_name}  seed={seed}"
        )

        train_df, val_df, test_df = split_three_way(
            df, *fracs, stratify=stratify, random_state=seed
        )
        print(
            f"  tamaños: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )

        # Persistir splits (útil para auditoría/reproducibilidad)
        case_split_dir = SPLITS_DIR / case_id
        case_split_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(case_split_dir / "train.csv", index=False)
        val_df.to_csv(case_split_dir / "val.csv", index=False)
        test_df.to_csv(case_split_dir / "test.csv", index=False)

        ref = to_evidently(train_df)
        cur_val = to_evidently(val_df)
        cur_test = to_evidently(test_df)

        val_html = REPORTS_DIR / f"{case_id}__val.html"
        test_html = REPORTS_DIR / f"{case_id}__test.html"

        val_summary = run_drift_report(ref, cur_val, val_html, label="val")
        test_summary = run_drift_report(ref, cur_test, test_html, label="test")

        rows.append(
            {
                "case_id": case_id,
                "stratify": stratify,
                "ratio": ratio_name,
                "seed": seed,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "drift_share_val": val_summary["share_drift"],
                "drift_share_test": test_summary["share_drift"],
                "n_drifted_val": val_summary["n_drifted"],
                "n_drifted_test": test_summary["n_drifted"],
                "n_total_cols": val_summary["n_total"],
                "drift_per_col_val": val_summary["per_column_drift"],
                "drift_per_col_test": test_summary["per_column_drift"],
                "report_val": val_summary["html"],
                "report_test": test_summary["html"],
            }
        )
        print()

    return rows


# ---------------------------------------------------------------------------
# Tabla resumen
# ---------------------------------------------------------------------------
def write_summary(rows: list[dict]) -> None:
    df_summary = pd.DataFrame(
        [
            {
                "case_id": r["case_id"],
                "stratify": r["stratify"],
                "ratio": r["ratio"],
                "seed": r["seed"],
                "n_train": r["n_train"],
                "n_val": r["n_val"],
                "n_test": r["n_test"],
                "drift_share_val": r["drift_share_val"],
                "drift_share_test": r["drift_share_test"],
                "n_drifted_val": r["n_drifted_val"],
                "n_drifted_test": r["n_drifted_test"],
                "n_total_cols": r["n_total_cols"],
            }
            for r in rows
        ]
    )

    csv_path = DOCS_DIR / "summary.csv"
    md_path = DOCS_DIR / "summary_table.md"
    json_path = DOCS_DIR / "summary_full.json"

    df_summary.to_csv(csv_path, index=False)

    # Tabla en Markdown lista para pegar en el documento de discusión
    md_lines = [
        "| Estratif. | Ratio | Semilla | n_train | n_val | n_test | "
        "Frac. drift VAL | Frac. drift TEST |",
        "|:---:|:---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df_summary.iterrows():
        md_lines.append(
            f"| {'sí' if r['stratify'] else 'no'} "
            f"| {r['ratio']} "
            f"| {r['seed']} "
            f"| {r['n_train']} "
            f"| {r['n_val']} "
            f"| {r['n_test']} "
            f"| {r['drift_share_val']:.2%} "
            f"| {r['drift_share_test']:.2%} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    print(f"Resumen escrito en:\n  - {csv_path}\n  - {md_path}\n  - {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No encuentro el dataset en {DATA_PATH}. "
            "Coloca el CSV crudo del Titanic ahí."
        )

    rows = run_all()
    write_summary(rows)
    print("\nDone.")
