#!/usr/bin/env bash
# deploy_all.sh — Starts Prefect infrastructure, uploads the Titanic dataset
# to W&B, and executes both the coupled and decoupled pipelines end-to-end.
# Usage:  bash deploy_all.sh
set -euo pipefail

# ── Resolve project root (always run from here) ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

echo "════════════════════════════════════════════════════"
echo " Step 1 — Starting Prefect infrastructure (Docker)"
echo "════════════════════════════════════════════════════"
docker compose -f docker/docker-compose.yml up -d
echo "Waiting for Prefect server to become healthy..."
until curl -sf http://localhost:4200/api/health > /dev/null 2>&1; do
    echo "  ... not ready yet, retrying in 5s"
    sleep 5
done
echo "Prefect server is ready at http://localhost:4200"

echo ""
echo "════════════════════════════════════════════════════"
echo " Step 2 — Uploading raw Titanic dataset to W&B"
echo "════════════════════════════════════════════════════"
python wandb_init.py

echo ""
echo "════════════════════════════════════════════════════"
echo " Step 3 — Running COUPLED flow (monolith)"
echo "════════════════════════════════════════════════════"
python -c "
from flows.coupled_flow.flow import coupled_pipeline
coupled_pipeline()
"

echo ""
echo "════════════════════════════════════════════════════"
echo " Step 4 — Running DECOUPLED Flow A (data pipeline)"
echo "════════════════════════════════════════════════════"
python -c "
from flows.decoupled_flows.flow_a_data import data_pipeline
data_pipeline()
"

echo ""
echo "════════════════════════════════════════════════════"
echo " Step 5 — Running DECOUPLED Flow B (training pipeline)"
echo "════════════════════════════════════════════════════"
python -c "
from flows.decoupled_flows.flow_b_training import training_pipeline
training_pipeline()
"

echo ""
echo "════════════════════════════════════════════════════"
echo " All flows completed successfully!"
echo " Prefect UI  → http://localhost:4200"
echo " W&B project → https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "════════════════════════════════════════════════════"
