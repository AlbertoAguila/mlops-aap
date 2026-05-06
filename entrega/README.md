# Entrega — Titanic · Análisis de deriva con EvidentlyAI

Este directorio contiene **todo lo necesario para evaluar la práctica**: el
informe de discusión, los 24 reports de Evidently, el código que los generó
y los datos auditables de respaldo.

## Qué contiene cada carpeta

```
entrega/
├── README.md           # este archivo
├── scripts/            # código que genera todo
├── data/splits/        # los 12 splits train/val/test guardados como CSV
├── reports/            # los 24 reports HTML de Evidently (val + test × 12 casos)
└── resultados/         # informe + tablas resumen
```

A continuación explico una a una, en lenguaje claro, qué hay en cada sitio.

### `scripts/`

Una sola pieza: **`run_titanic_drift.py`**. Es el código Python que
automatiza el experimento. Funciones reutilizables para cargar el dataset,
dividirlo en train/val/test bajo cualquier combinación de
estratificación/ratio/semilla, y un bucle que recorre las 12 condiciones
generando los splits en CSV, los 24 reports HTML y la tabla resumen. Si
quieres reproducir el experimento, este es el único archivo que necesitas
ejecutar.

### `data/splits/`

12 subcarpetas (una por caso experimental) con los archivos
`train.csv`, `val.csv` y `test.csv` que efectivamente se usaron. Están aquí
por **trazabilidad**: si alguien duda de los resultados, puede recoger un
split concreto y replicar el análisis sin tener que volver a sortear nada.

Convención de nombres de las subcarpetas:
`<estratif>_<ratio>_seed<semilla>` → por ejemplo `strat_60-20-20_seed42`.

### `reports/`

Los **24 reports HTML** de EvidentlyAI, dos por cada uno de los 12 casos:
uno comparando train contra val y otro comparando train contra test. Cada
HTML muestra el dashboard interactivo con la deriva por columna, los
estadísticos usados y los umbrales aplicados.

Convención de nombres:
`<estratif>_<ratio>_seed<semilla>__<conjunto>.html`

- `<estratif>` ∈ `strat`, `nostrat`
- `<ratio>` ∈ `60-20-20`, `90-05-05`, `98-01-01`
- `<semilla>` ∈ `42`, `7`
- `<conjunto>` ∈ `val`, `test`

Ejemplos:

- `strat_60-20-20_seed42__val.html` — estratificado, 60/20/20, semilla 42, val vs train.
- `nostrat_98-01-01_seed7__test.html` — sin estratificar, 98/1/1, semilla 7, test vs train.

### `resultados/`

Aquí está **el informe** y los datos que lo respaldan. Cuatro archivos:

| Archivo | Para qué sirve | ¿Lo abre la profe? |
|---------|----------------|---------------------|
| `discusion.md` | El informe principal con la tabla resumen, análisis por parámetro y conclusiones | **Sí**, esto es lo que se lee |
| `summary.csv` | La misma tabla resumen pero en CSV, abrible con Excel | Tal vez, si quiere copiar datos |
| `summary_table.md` | Borrador autogenerado por el script — la tabla suelta sin discusión | No, es intermedio |
| `summary_full.json` | Datos crudos por caso (qué columna concreta drifteó, cuántas se midieron, etc.) | No, es para máquina |

Detalle de cada uno:

- **`discusion.md`** — Documento principal en Markdown. Tiene introducción,
  metodología, tabla resumen, 6 secciones de discusión (qué significa drift
  aquí, efecto de la estratificación, del ratio, de la semilla, diferencias
  val/test, causas) y una conclusión con 5 recomendaciones. Si tuvieras
  que entregar **un solo archivo**, sería este.
- **`summary.csv`** — La tabla resumen exportada para hoja de cálculo.
  Útil si quieres reusar los números o pintar gráficas. 12 filas (una por
  caso) con 12 columnas (`case_id`, `stratify`, `ratio`, `seed`, `n_train`,
  `n_val`, `n_test`, `drift_share_val`, `drift_share_test`, `n_drifted_val`,
  `n_drifted_test`, `n_total_cols`).
- **`summary_table.md`** — La misma tabla en Markdown, sin la discusión
  alrededor. La genera el script automáticamente, está pensada para
  pegarse en otro documento si hace falta. Borrarla no perdería información
  porque ya aparece dentro de `discusion.md`.
- **`summary_full.json`** — JSON estructurado con el detalle completo por
  caso: qué reporte HTML lo respalda, cuántas columnas se midieron, las
  fracciones de drift y, cuando es posible, la lista exacta de columnas con
  drift. Es la prueba auditable de los números del informe.

## Cómo reproducir todo desde cero

Desde la carpeta `lab A/`:

```bash
uv add scikit-learn          # solo la primera vez
uv run entrega/scripts/run_titanic_drift.py
```

(O con pip + venv si no usas uv: ver instrucciones en `lab A/README.md`.)

El script:

1. Carga `lab A/titanic-dataset.csv`.
2. Recorre las 12 condiciones experimentales (2 estratificación × 3 ratios × 2 semillas).
3. Para cada caso divide en train/val/test, escribe los CSV en `data/splits/`
   y dos reports HTML de Evidently en `reports/` (val vs train, test vs train).
4. Construye `resultados/summary.csv` y `resultados/summary_table.md` con
   la fracción de columnas con drift por caso.

## Esquema de columnas usado

Para el análisis de drift se conservaron 8 columnas del Titanic:

- **Numéricas:** `Age`, `Fare`, `SibSp`, `Parch`
- **Categóricas:** `Survived` (target), `Pclass`, `Sex`, `Embarked`
- **Descartadas:** `PassengerId`, `Name`, `Ticket`, `Cabin`
  (identificadores, texto libre o demasiados NaN)

Por eso, en cada caso, una fracción de drift de 0,125 = 1 columna sobre 8.
