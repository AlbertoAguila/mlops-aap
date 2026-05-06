# Entrega — Titanic · Análisis de deriva con EvidentlyAI

Estructura del entregable:

```
entrega/
├── scripts/
│   └── run_titanic_drift.py        # genera splits + reports HTML + tabla resumen
├── data/
│   └── splits/<case_id>/           # train/val/test.csv por cada caso (auditable)
├── reports/
│   └── <case_id>__val.html         # 12 reports val + 12 reports test = 24 HTML
│   └── <case_id>__test.html
└── docs/
    ├── summary.csv                 # tabla cruda con todos los campos
    ├── summary_table.md            # tabla resumen lista para pegar
    ├── summary_full.json           # detalle drift por columna en cada caso
    └── discusion.md                # documento de discusión
```

## Cómo ejecutar

Desde la carpeta `lab A/`:

```bash
uv add scikit-learn          # solo la primera vez
uv run entrega/scripts/run_titanic_drift.py
```

El script:

1. Carga `lab A/titanic-dataset.csv`.
2. Recorre las 12 condiciones experimentales (2 estratificación × 3 ratios × 2 semillas).
3. Para cada caso divide en train/val/test, escribe los CSV de splits y dos
   reports HTML de Evidently (val vs train, test vs train).
4. Construye `docs/summary.csv` y `docs/summary_table.md` con la fracción de
   columnas con drift por caso.

## Convención de nombres

`<case_id> = <strat|nostrat>_<ratio>_seed<seed>`

Ejemplos:

- `strat_60-20-20_seed42__val.html`
- `nostrat_98-01-01_seed7__test.html`

## Columnas del esquema usado

- Numéricas: `Age`, `Fare`, `SibSp`, `Parch`
- Categóricas: `Survived` (target), `Pclass`, `Sex`, `Embarked`
- Descartadas: `PassengerId`, `Name`, `Ticket`, `Cabin`
  (identificadores / texto libre / muy alta tasa de NaN)

## Tras ejecutar el script

1. Abre `docs/summary_table.md` y copia el contenido.
2. Pega esa tabla en `docs/discusion.md` reemplazando el bloque entre
   `<!-- BEGIN summary_table.md -->` y `<!-- END summary_table.md -->`.
3. Los 24 HTML quedan en `reports/`, listos para subir a Moodle junto al
   resto de la entrega.
