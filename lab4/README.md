# Lab 4 – MLflow Models & Registry

## Dataset

Se ha utilizado el dataset **PIMA Diabetes Dataset**, que contiene información médica de pacientes y permite predecir si un paciente tiene diabetes.

La variable objetivo es **Outcome**.

---

## Experimento de prueba

En primer lugar se realizó un experimento inicial para comprobar el funcionamiento de MLflow:

1. Entrenamiento de un modelo Random Forest.
2. Registro del modelo en MLflow.
3. Registro en el Model Registry.
4. Exposición del modelo mediante una API con `mlflow models serve`.
5. Verificación de predicciones mediante llamadas a la API.

Esto permitió validar el flujo básico de trabajo con MLflow.

---

## Experimentos completos

Posteriormente se realizaron varios experimentos utilizando diferentes combinaciones de hiperparámetros del modelo Random Forest.

Los hiperparámetros explorados fueron:

- `n_estimators`
- `max_depth`

Para ello se utilizó el script `grid_search_experiments.py`.

Cada combinación de parámetros fue registrada como un **run independiente en MLflow**, guardando las siguientes métricas:

- Accuracy  
- Precision  
- Recall  
- F1 Score

---

## Comparación de modelos

| Modelo | n_estimators | max_depth | Accuracy | Precision | Recall | F1 |
|------|------|------|------|------|------|------|
| indecisive-finch-108 | 200 | 10 | 0.7597 | 0.6607 | **0.6727** | 0.6667 |
| bustling-doe-751 | 150 | 7 | 0.7727 | 0.6923 | 0.6545 | **0.6729** |
| flawless-bat-833 | 100 | 5 | **0.7792** | **0.7234** | 0.6182 | 0.6667 |
| respected-ram-25 | 50 | 3 | 0.7467 | 0.6818 | 0.5454 | 0.6061 |

---

## Selección del modelo

En problemas médicos como la detección de diabetes, la métrica **recall** es especialmente importante.

Un **falso negativo** significa que un paciente con la enfermedad no es detectado por el modelo, lo cual puede tener consecuencias médicas importantes.

Por este motivo se seleccionó el modelo **indecisive-finch-108**, que presenta el mayor valor de recall entre los modelos evaluados.

---

## Exposición del modelo mediante API

El modelo seleccionado fue expuesto mediante la API de MLflow utilizando `mlflow models serve`.  
Esto permitió realizar predicciones mediante peticiones HTTP.

---

## Evaluación final mediante API

Se implementó el script `evaluate_api.py`, que envía las observaciones del dataset de test a la API y obtiene las predicciones del modelo.

Las métricas obtenidas fueron:

- Accuracy: 0.727  
- Precision: 0.571  
- Recall: 0.64  
- F1 Score: 0.604  

---

## Conclusiones

Los experimentos realizados muestran que el modelo Random Forest es capaz de detectar patrones relevantes en el dataset de diabetes.

Aunque algunos modelos presentan mayor accuracy o precision, se priorizó el recall debido al contexto médico del problema.

El modelo seleccionado ofrece un equilibrio razonable entre las métricas evaluadas y permite detectar una proporción adecuada de pacientes con diabetes.

En trabajos futuros se podría mejorar el rendimiento utilizando:

- técnicas de balanceo de clases  
- optimización de hiperparámetros más extensa  
- modelos adicionales como Gradient Boosting o XGBoost.