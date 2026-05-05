#!/usr/bin/env bash
set -euo pipefail

BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:7860"
HEALTH_TIMEOUT=60
POLL_INTERVAL=2

# ── Colores / prefijos ────────────────────────────────────────
info()  { echo "[INFO]  $*"; }
ok()    { echo "[OK]    $*"; }
error() { echo "[ERROR] $*"; }

# ── 1. Esperar a que /health devuelva HTTP 200 ────────────────
info "Esperando a que el backend esté listo (máx. ${HEALTH_TIMEOUT}s)..."
elapsed=0
while true; do
    status=$(curl -s -o /dev/null -w "%{http_code}" "${BACKEND_URL}/health" 2>/dev/null || echo "000")
    if [ "${status}" = "200" ]; then
        ok "Backend healthy (${BACKEND_URL}/health respondió 200 en ${elapsed}s)"
        break
    fi
    if [ "${elapsed}" -ge "${HEALTH_TIMEOUT}" ]; then
        error "Timeout: /health no respondió 200 en ${HEALTH_TIMEOUT}s (último código: ${status})"
        exit 1
    fi
    info "  /health → ${status} — reintentando en ${POLL_INTERVAL}s... (${elapsed}s transcurridos)"
    sleep "${POLL_INTERVAL}"
    elapsed=$((elapsed + POLL_INTERVAL))
done

# ── 2. POST /predict con muestra Iris setosa ──────────────────
info "Enviando predicción de prueba a ${BACKEND_URL}/predict..."
PREDICT_RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" \
    -X POST "${BACKEND_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}')

PREDICT_BODY=$(echo "${PREDICT_RESPONSE}" | sed '/HTTP_CODE:/d')
PREDICT_CODE=$(echo "${PREDICT_RESPONSE}" | grep "HTTP_CODE:" | cut -d: -f2)

if [ "${PREDICT_CODE}" = "200" ]; then
    ok "POST /predict → HTTP ${PREDICT_CODE}"
    echo "    Respuesta: ${PREDICT_BODY}"
else
    error "POST /predict → HTTP ${PREDICT_CODE}"
    echo "    Respuesta: ${PREDICT_BODY}"
    exit 1
fi

# ── 3. Comprobar que el frontend responde HTTP 200 ────────────
info "Comprobando frontend en ${FRONTEND_URL}..."
FRONTEND_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${FRONTEND_URL}" 2>/dev/null || echo "000")
if [ "${FRONTEND_CODE}" = "200" ]; then
    ok "Frontend accesible (${FRONTEND_URL} → HTTP ${FRONTEND_CODE})"
else
    error "Frontend no responde correctamente (${FRONTEND_URL} → HTTP ${FRONTEND_CODE})"
    exit 1
fi

# ── Resultado final ───────────────────────────────────────────
echo ""
echo "========================================="
echo "  STACK OK"
echo "========================================="
exit 0
