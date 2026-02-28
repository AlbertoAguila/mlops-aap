# LAB 3 – Respuestas a las Preguntas de Reflexión

## Pregunta 1 – Reproducibilidad del modelo bootstrap

Si se elimina la carpeta `models/` y se vuelve a arrancar el servidor, el sistema crea un nuevo modelo base mediante bootstrap.

La reproducibilidad depende de si el código fija una semilla (`random_state`) tanto en el modelo como en la partición train/test.  
Si se utiliza `random_state` fijo, el modelo generado y su accuracy serán reproducibles entre ejecuciones.  
Si no se fija, pueden producirse pequeñas variaciones debido a la aleatoriedad en la división de datos o en el proceso de optimización.

Por tanto, la línea o parámetro clave para garantizar reproducibilidad es el uso de `random_state` en el modelo y en el `train_test_split`.

---

## Pregunta 2 – Dataset con una sola clase

Si se envían 10 muestras correctamente etiquetadas pero todas pertenecen a la misma clase (por ejemplo, setosa), el modelo no debería entrenarse correctamente.

Un clasificador multiclase necesita al menos dos clases distintas para aprender fronteras de decisión.  
El endpoint `/train` valida que haya mínimo 2 clases distintas en el conjunto de entrenamiento.

En este caso, la API debería:
1. Detectar que solo hay una clase.
2. Rechazar el entrenamiento.
3. Mantener el modelo anterior activo.

Esta restricción tiene sentido porque evita entrenar modelos inválidos o sin capacidad real de generalización.

---

## Pregunta 3 – Registro de modelos rechazados

El sistema registra en el historial tanto los modelos activados como los rechazados.

Esta decisión es importante por motivos de auditoría y trazabilidad:

- Permite identificar qué datos degradaron el modelo.
- Facilita analizar cuándo y por qué se intentó un reentrenamiento fallido.
- Proporciona información para debugging y mejora del pipeline.

Si solo se guardaran los modelos activados, se perdería información crítica sobre intentos fallidos de mejora.

---

## Pregunta 4 – Uso de >= en el gate de calidad

El gate actual utiliza la condición:

accuracy_nuevo >= accuracy_anterior

Esto implica que un modelo con exactamente la misma accuracy que el anterior se activa.

Usar `>=` puede ser razonable cuando:

- El dataset crece progresivamente con datos de calidad.
- La métrica se estabiliza cerca de su valor máximo.
- Puede haber mejoras en aspectos no capturados por la accuracy global.

Si se utilizara un `>` estricto, el sistema podría bloquear actualizaciones legítimas cuando la accuracy se mantiene constante debido a saturación de la métrica.

Por tanto, `>=` ofrece mayor flexibilidad en escenarios reales donde la mejora no siempre se refleja en un aumento numérico visible en la métrica principal.
