# LAB 3 — `app-iris-ct`: Continuous Training con FastAPI

## Estructura del proyecto

```
app-iris-ct/
├── main.py              ← servidor FastAPI extendido  (a desarrollar)
├── demo_ct.py           ← script de demostración del flujo CT
├── Dockerfile
├── requirements.txt
├── models/              ← creada automáticamente en runtime
│   ├── model_active.joblib       ← modelo activo serializado
│   ├── accumulated_data.joblib   ← dataset acumulado entre entrenamientos
│   └── training_history.json    ← registro de versiones
└── README.md
```
# LAB 3 – Continuous Training API (Iris)

## Descripción

En este laboratorio se implementa una API basada en FastAPI que permite:

- Servir predicciones sobre el dataset Iris
- Reentrenar el modelo con nuevas muestras etiquetadas
- Registrar el historial completo de versiones
- Activar o rechazar modelos automáticamente según su accuracy

El sistema implementa un mecanismo de control (gate) que solo activa un nuevo modelo si su accuracy es mayor o igual que la del modelo activo anterior.

---

## Ejercicio 2 – Análisis del historial de entrenamiento

Tras ejecutar la demo y realizar entrenamientos adicionales, el archivo `training_history.json` contiene múltiples versiones del modelo.

Se verificó que:

- Existen al menos 4 versiones registradas.
- Existen al menos 2 modelos rechazados.
- Cada entrenamiento queda registrado independientemente de si se activa o no.

El sistema demuestra trazabilidad completa del ciclo de vida del modelo, incluyendo:

- Versión
- Fecha de entrenamiento
- Accuracy
- Número de muestras
- Estado (activado / rechazado)

Esto garantiza control y auditabilidad del proceso de continuous training.

---

## Ejercicio 3 – Comparación retrain_from_scratch

Se realizaron dos llamadas al endpoint `/train` utilizando exactamente las mismas 10 muestras correctamente etiquetadas.

### Caso A – retrain_from_scratch = false

- Se acumuló el histórico previo.
- Accuracy obtenida: 1.0
- Modelo activado.

### Caso B – retrain_from_scratch = true

- Se entrenó únicamente con las 10 muestras enviadas.
- Accuracy obtenida: 1.0
- Modelo activado.

### Análisis

En esta ejecución concreta no se observaron diferencias en rendimiento debido a que las muestras eran coherentes y correctamente etiquetadas.

Conceptualmente:

- `false` proporciona mayor estabilidad al aprovechar datos históricos acumulados.
- `true` permite descartar datos antiguos potencialmente corruptos.

Esto demuestra que la API permite tanto aprendizaje incremental como reentrenamiento completo bajo control de calidad basado en métricas.

---

## Tecnologías utilizadas

- FastAPI
- Scikit-learn
- Uvicorn
- Python 3.12

---

## Ejecución

Activar entorno virtual:

    source .venv/bin/activate

Lanzar servidor:

    uvicorn main:app --reload

Documentación Swagger:

    http://127.0.0.1:8000/docs
---

## Endpoints

| Método | Ruta               | Descripción                                              |
| ------- | ------------------ | --------------------------------------------------------- |
| GET     | `/health`        | Estado del servicio y versión del modelo activo          |
| POST    | `/predict`       | Predicción con el modelo activo (igual que `app-iris`) |
| POST    | `/train`         | Reentrenamiento con nuevas muestras etiquetadas           |
| GET     | `/model/info`    | Metadata del modelo activo e historial de versiones       |
| DELETE  | `/model/history` | Resetea el historial (para pruebas)                       |

### Esquema de `/train` (request)

```json
{
  "samples": [
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2,
      "label": 0
    }
  ],
  "retrain_from_scratch": false
}
```

- `label` acepta `0` (setosa), `1` (versicolor) o `2` (virginica).
- Se requieren **mínimo 5 muestras** por request.
- `retrain_from_scratch: false` → las nuevas muestras se **acumulan** al dataset anterior.
- `retrain_from_scratch: true` → se entrena **solo** con las muestras enviadas.

### Esquema de `/train` (response)

```json
{
  "status": "activado",
  "model_version": "v2.0-a3f9b1",
  "accuracy_new": 0.9667,
  "accuracy_previous": 0.9333,
  "model_updated": true,
  "message": "Nuevo modelo activado. Accuracy 0.9667 >= anterior (0.9333)"
}
```

### Esquema de `/model/info` (response)

```json
{
  "active_version": "v2.0-a3f9b1",
  "trained_at": "2024-11-15T10:23:44Z",
  "accuracy": 0.9667,
  "n_training_samples": 120,
  "algorithm": "LogisticRegression",
  "history": [
    {
      "version": "v1.0-base",
      "trained_at": "2024-11-15T09:00:00Z",
      "accuracy": 0.9333,
      "n_training_samples": 120,
      "source": "bootstrap (iris dataset completo)",
      "activated": true
    },
    {
      "version": "v2.0-a3f9b1",
      "trained_at": "2024-11-15T10:23:44Z",
      "accuracy": 0.9667,
      "n_training_samples": 140,
      "source": "incremental (+20 muestras nuevas, 120 anteriores)",
      "activated": true
    }
  ]
}
```

---

## Lógica del gate de calidad

```
accuracy_nuevo >= accuracy_anterior  →  ACTIVAR y guardar modelo
accuracy_nuevo <  accuracy_anterior  →  RECHAZAR, mantener modelo anterior
```

El registro del intento de entrenamiento **siempre** queda en el historial, incluso si el modelo es rechazado. Esto permite auditar qué datos degradaron el modelo.

---

## Instrucciones de desarrollo

### Prerrequisitos

```bash
pip install -r requirements.txt
```

### Arrancar el servidor en local

```bash
uvicorn main:app --reload
```

Accede a la documentación interactiva en: [http://localhost:8000/docs](http://localhost:8000/docs)

### Arrancar con Docker

```bash
# Construir imagen
docker build -t iris-ct .

# Lanzar contenedor con volumen para persistir los modelos
docker run -d \
  -p 8000:80 \
  -v iris-ct-models:/app/models \
  --name iris-ct \
  iris-ct

docker run -d -p 8000:80 -v iris-ct-models:/app/models --name iris-ct   iris-ct
```

Con el volumen `-v iris-ct-models:/app/models` los modelos entrenados **sobreviven** al reinicio del contenedor.

### Ejecutar la demo completo

```bash
# Servidor debe estar arrancado primero
python demo_ct.py --reset
```

El flag `--reset` restaura el modelo base antes de ejecutar el flujo. Salida esperada:

```
════════════════════════════════════════════════════════════
  🌸  DEMO: Iris Continuous Training API
════════════════════════════════════════════════════════════
  Host: http://localhost:8000

────────────────────────────────────────────────────────────
  PASO 0 – Health check
────────────────────────────────────────────────────────────
  ✅  Servicio activo. Modelo activo: v1.0-base

  [...]

  PASO 2 – Reentrenamiento con muestras CORRECTAS
────────────────────────────────────────────────────────────
  ✅  Nuevo modelo ACTIVADO → versión: v2.0-xxxx
  ✅  Accuracy nuevo: 0.9667  |  Anterior: 0.9333

────────────────────────────────────────────────────────────
  PASO 3 – Reentrenamiento con muestras RUIDOSAS
────────────────────────────────────────────────────────────
  ✅  Modelo RECHAZADO como esperado. Accuracy 0.6000 < anterior 0.9667
```

---

## Ejercicios de la actividad

**Ejercicio 1 — Reproducir el workflow**
Arranca el servidor, ejecuta `demo_ct.py --reset` y comprueba que los 4 pasos funcionan correctamente. Captura la salida completa del terminal y el JSON de `/model/info` al final.

**Ejercicio 2 — Explorar el historial**

Tras ejecutar la demo, abre `models/training_history.json` y responde:

- ¿Cuántas versiones se han registrado?
- ¿Qué versiones fueron activadas y cuáles rechazadas?
- ¿Por qué el modelo entrenado con muestras ruidosas fue rechazado?

NOTA: Los ficheros generados están dentro del contenedor, montados en un volumen: `iris-ct-models:/app/models`

**Ejercicio 3 — Entrenamiento incremental vs. desde cero**

Realiza dos llamadas a `/train` con el mismo conjunto de 10 muestras:

- Primera vez con `retrain_from_scratch: false`
- Segunda vez con `retrain_from_scratch: true`

¿Observas diferencias en el accuracy? ¿Por qué?

---


## Referencias

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [Google MLOps: Continuous delivery and automation pipelines in ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Joblib documentation](https://joblib.readthedocs.io/)
