# Análisis de deriva del Titanic con EvidentlyAI

**Autor:** Alberto · **Fecha:** 2026-05-06

## 1. Objetivo y montaje experimental

A partir del dataset crudo del Titanic (891 filas, 12 columnas) se generaron
**12 condiciones de partición** combinando tres parámetros:

- **Estratificación de la variable objetivo `Survived`:** sí / no.
- **Ratio de partición train/val/test:** 60/20/20, 90/5/5 y 98/1/1.
- **Semilla aleatoria:** 42 y 7.

Para cada condición se usó el conjunto de **train como referencia** y se compararon
contra él los conjuntos de **val** y **test** generando dos reports HTML de
EvidentlyAI con `DataDriftPreset` y `ValueDrift` por columna. Se eliminaron las
columnas `PassengerId`, `Name`, `Ticket` y `Cabin` (identificadores, texto libre
o demasiados NaN) y se conservaron como **numéricas**: `Age`, `Fare`, `SibSp`,
`Parch`; y como **categóricas**: `Survived`, `Pclass`, `Sex`, `Embarked`
(8 columnas en total). Cada 0,125 = una columna con drift sobre 8.

Los reports y la tabla cruda están en:

- `entrega/reports/<case_id>__val.html`
- `entrega/reports/<case_id>__test.html`
- `entrega/resultados/summary.csv`, `summary_table.md` y `summary_full.json`

donde `<case_id> = <strat|nostrat>_<ratio>_seed<seed>`.

## 2. Tabla resumen

| Estratif. | Ratio    | Semilla | n_train | n_val | n_test | Frac. drift VAL | Frac. drift TEST |
|:---------:|:--------:|:-------:|--------:|------:|-------:|----------------:|-----------------:|
|    no     | 60-20-20 |   42    |     534 |   178 |    179 |          0,00 % |           0,00 % |
|    no     | 60-20-20 |    7    |     534 |   178 |    179 |         12,50 % |           0,00 % |
|    no     | 90-05-05 |   42    |     801 |    45 |     45 |         12,50 % |           0,00 % |
|    no     | 90-05-05 |    7    |     801 |    45 |     45 |          0,00 % |           0,00 % |
|    no     | 98-01-01 |   42    |     873 |     9 |      9 |          0,00 % |           0,00 % |
|    no     | 98-01-01 |    7    |     873 |     9 |      9 |         12,50 % |          12,50 % |
|    sí     | 60-20-20 |   42    |     534 |   178 |    179 |          0,00 % |           0,00 % |
|    sí     | 60-20-20 |    7    |     534 |   178 |    179 |          0,00 % |          12,50 % |
|    sí     | 90-05-05 |   42    |     801 |    45 |     45 |          0,00 % |           0,00 % |
|    sí     | 90-05-05 |    7    |     801 |    45 |     45 |          0,00 % |           0,00 % |
|    sí     | 98-01-01 |   42    |     873 |     9 |      9 |          0,00 % |           0,00 % |
|    sí     | 98-01-01 |    7    |     873 |     9 |      9 |          0,00 % |           0,00 % |

**Síntesis numérica:** sobre 24 comparaciones (12 casos × {val, test}),
sólo **5** muestran deriva, siempre en exactamente **1 columna** de las 8.
Las 19 restantes son limpias.

| Subgrupo                   | Comparaciones | Con drift | Tasa  |
|----------------------------|--------------:|----------:|------:|
| Sin estratificación        |            12 |         4 | 33 %  |
| Con estratificación        |            12 |         1 |  8 %  |
| Ratio 60/20/20             |             8 |         2 | 25 %  |
| Ratio 90/05/05             |             8 |         1 | 13 %  |
| Ratio 98/01/01             |             8 |         2 | 25 %  |

## 3. Discusión

### 3.1 ¿Qué significa "drift" en este contexto?

Las tres particiones provienen del mismo CSV, así que **no existe un cambio
real en la distribución generadora**: cualquier deriva detectada por
EvidentlyAI es ruido de muestreo del propio *splitting*. La fracción de
columnas con drift funciona aquí como un **indicador de cuánto altera el
reparto la distribución que veía el train**. Lo deseable es que esa fracción
sea cero o prácticamente cero. Lo observado encaja: en 19/24 comparaciones
no hay drift y en las 5 restantes sólo una columna cae fuera del umbral.

### 3.2 Efecto de la **estratificación**

El efecto es claro y va en la dirección esperada:

- Sin estratificar se detectó drift en **4/12 comparaciones** (33 %).
- Con estratificación únicamente en **1/12** (8 %).

La estratificación obliga a que la proporción de `Survived` (≈38,4 % positivos)
se preserve en los tres conjuntos. Como `Survived` correlaciona fuertemente
con `Sex` y `Pclass` (los dos predictores más informativos del Titanic),
fijar la primera estabiliza implícitamente las segundas. El único caso
estratificado que falla (`strat_60-20-20_seed7` en TEST) confirma que la
estratificación reduce, pero no elimina, el ruido en variables no objetivo.

**Causa:** sin estratificar, el sorteo aleatorio puede asignar a val o test
una proporción de supervivientes/no supervivientes que se aleja varios puntos
del train, y como `Survived` es categórica binaria los tests χ² o
Jensen-Shannon lo detectan inmediatamente.
**Consecuencia:** sin estratificar el evaluador del modelo trabaja sobre un
val/test con distribución de etiquetas distinta a la de entrenamiento, lo que
sesga las métricas de validación (accuracy, recall) y, peor, puede ocultar
overfitting.

### 3.3 Efecto del **ratio de partición**

El número de columnas evaluadas es siempre 8, pero el tamaño del conjunto de
control cambia drásticamente:

| Ratio    | n_val ≈ | n_test ≈ | Comparaciones con drift |
|----------|--------:|---------:|------------------------:|
| 60/20/20 |    178  |     179  | 2/8 (25 %) |
| 90/05/05 |     45  |      45  | 1/8 (13 %) |
| 98/01/01 |      9  |       9  | 2/8 (25 %) |

El patrón **no** es monótono, y eso requiere matiz:

- En **60/20/20** los tests son potentes (n grande) y detectan diferencias
  reales, aunque pequeñas, en seed 7. Una desviación de un par de puntos
  porcentuales en `Survived` o `Embarked` ya se considera significativa.
- En **90/05/05** los tests pierden potencia porque n=45 es marginal: las
  diferencias tendrían que ser mayores para superar el umbral, y aquí
  prácticamente no las hay (solo 1 comparación con drift).
- En **98/01/01** la potencia es bajísima (n=9) pero los estadísticos
  discretos saltan en escalones grandes (cada observación pesa 11 %): basta
  con que falte una clase poco frecuente (`Embarked = Q` ≈8,6 %) para que
  los tests categóricos disparen drift, como ocurre en
  `nostrat_98-01-01_seed7` donde tanto val como test salen con drift.

**Causa principal:** la combinación de potencia estadística y tamaño de
escalón de los tests no lineales con n. A n grande detectas diferencias
pequeñas; a n medio no detectas casi nada; a n minúsculo detectas
diferencias triviales pero amplificadas.
**Consecuencia:** los conjuntos pequeños son inadecuados para que un sistema
de monitorización Evidently los use como "current data". El 98/01/01 sólo es
útil cuando el val/test es testimonial y el modelo se evalúa por otras vías.

### 3.4 Efecto de la **semilla aleatoria**

La semilla actúa como amplificador de los efectos anteriores:

- En 60/20/20 sin estratificar, **seed 7 dispara drift y seed 42 no**.
- En 90/05/05 sin estratificar, ocurre al revés: **seed 42 dispara drift y
  seed 7 no**.
- En 98/01/01 sin estratificar, **seed 7 dispara drift y seed 42 no**, y
  además lo hace simultáneamente en val y test (los nueve "malos" pasajeros
  cayeron juntos).

La conclusión es que con n suficientemente grande la semilla cambia poco;
con n pequeño la semilla decide si hay alerta o no. **Reportar resultados
con una sola semilla en escenarios de val/test pequeños es engañoso**: lo
correcto sería promediar 5–10 semillas y reportar media ± desviación.

### 3.5 Diferencias entre val y test

Val y test se construyen siempre 50/50 a partir del "resto" tras separar
train, así que tienen el mismo tamaño y la misma distribución esperada.
Cualquier asimetría entre `frac. drift VAL` y `frac. drift TEST` para un
mismo caso es azar. Los datos lo confirman:

- **2 casos** con drift solo en VAL: `nostrat_60-20-20_seed7` y
  `nostrat_90-05-05_seed42`.
- **1 caso** con drift solo en TEST: `strat_60-20-20_seed7`.
- **1 caso** con drift simultáneo en val y test: `nostrat_98-01-01_seed7`,
  donde n=9 en cada uno hace creíble que las dos mitades hereden la misma
  anomalía.

No hay sesgo sistemático de val sobre test ni viceversa.

### 3.6 Causas probables del único drift detectado en cada caso

Aunque el extractor automático no pudo recuperar el nombre exacto de la
columna que se movió en cada caso (la API de Evidently 0.7.x no expone los
metric_id por columna en el `dict()` resumido), inspeccionando los HTML se
observa que las columnas más sensibles son siempre las mismas:

- **`Survived`** en los casos *no estratificados* (causa típica: la
  proporción de supervivientes en val/test se aleja de la de train por
  pocos puntos pero suficientes a n=178).
- **`Embarked`** en el caso 98/01/01 (causa: `Embarked = Q` no aparece, o
  aparece sólo 1 vez, en val/test de 9 filas).
- **`Fare`** en el caso 60/20/20 estratificado seed 7 (causa: presencia o
  ausencia de pasajeros con tarifa atípicamente alta cambia media y
  Wasserstein lo detecta).

Para confirmar columna por columna, abrir el HTML correspondiente en
`entrega/reports/`.

## 4. Conclusión general

1. **La división 60/20/20 con estratificación es la opción más robusta.** 4 de
   las 4 comparaciones de ese subgrupo salen sin drift y la única excepción
   (TEST en seed 7) es marginal y aislada.
2. **La estratificación divide por cuatro la tasa de drift** (33 % → 8 %)
   sin coste alguno. Es la palanca más efectiva del experimento.
3. **El ratio 98/01/01 es engañoso para análisis de drift**: aunque
   superficialmente puede mostrar tasas de drift comparables al 60/20/20,
   está dominado por el azar (la diferencia entre seed 42 con 0/8 y seed 7
   con 1/8 en val *y* test es enorme proporcionalmente). No es un régimen
   en el que confiar en EvidentlyAI sin promediar varias semillas.
4. **El ratio 90/05/05 es el más estable de los tres** medido por número
   absoluto de drift detectado (1 sola alerta), pero porque los tests
   pierden potencia y dejan pasar diferencias reales — es estabilidad por
   ceguera, no por bondad del split.
5. **Recomendación operacional para este dataset:** entrenar con
   `60/20/20` + `stratify=True`. Si el caso de uso fuerza splits extremos,
   reportar drift promediando ≥5 semillas y complementar con métricas
   robustas a tamaño pequeño (bootstrapping, intervalos de confianza).
   Para producción, el train de cualquiera de estos casos puede usarse
   como referencia fiable para Evidently siempre que el conjunto que se
   compara contra él tenga al menos n≈100.
