#!/usr/bin/env bash
# =============================================================================
# deploy_bridge.sh — Start the Semantic Cache Bridge API on Port 8002
# =============================================================================
#
# Run this on the EC2 instance where the Semantic Cache is already deployed.
# This starts axiom_bridge.py as a background process managed by systemd.
#
# PREREQUISITES:
#   1. The existing semantic cache services (port 8000, 8001) are already running.
#   2. .env contains BRIDGE_API_KEY, CACHE_API_URL, and BRIDGE_PORT.
#   3. httpx is installed: pip install httpx==0.28.1
#   4. Port 8002 is open in the EC2 Security Group (TCP inbound).
#
# USAGE:
#   chmod +x deploy_bridge.sh
#   ./deploy_bridge.sh
#
# To check status after starting:
#   sudo systemctl status semantic-cache-bridge
#
# To view live logs:
#   journalctl -u semantic-cache-bridge -f
#
# To stop:
#   sudo systemctl stop semantic-cache-bridge
#
# NOTE — /v1/cache/store limitation:
#   The current implementation of /v1/cache/store calls the upstream /query
#   endpoint to trigger the Cache Privacy Classifier for privacy gating.
#   This means the upstream generates a placeholder LLM response internally,
#   which is discarded — only the classification/persistence decision matters.
#   A future enhancement is to add a dedicated POST /store endpoint to main.py
#   that accepts (prompt, response_text, embedding) directly, allowing the
#   LLM-generated response to be injected verbatim. Track this as:
#   GitHub Issue: "Add /store endpoint to main.py for direct cache injection"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="semantic-cache-bridge"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
VENV_PATH="${SCRIPT_DIR}/.venv"
APP_MODULE="axiom_bridge:app"
BIND_HOST="0.0.0.0"
BIND_PORT="8002"

# Resolve the uvicorn executable (prefer .venv, fall back to system PATH)
if [[ -f "${VENV_PATH}/bin/uvicorn" ]]; then
    UVICORN="${VENV_PATH}/bin/uvicorn"
    PYTHON="${VENV_PATH}/bin/python"
elif command -v uvicorn &>/dev/null; then
    UVICORN="$(command -v uvicorn)"
    PYTHON="$(command -v python3)"
else
    echo "[ERROR] uvicorn not found. Activate your virtualenv or install it first."
    exit 1
fi

echo "[BRIDGE] Installing/updating httpx..."
"${PYTHON}" -m pip install --quiet httpx==0.28.1

echo "[BRIDGE] Writing systemd service to ${SERVICE_FILE}..."

sudo tee "${SERVICE_FILE}" > /dev/null <<EOF
[Unit]
Description=Semantic Cache Bridge API (Port 8002)
Documentation=https://github.com/your-org/semantic-cache
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${SCRIPT_DIR}
EnvironmentFile=${SCRIPT_DIR}/.env
ExecStart=${UVICORN} ${APP_MODULE} --host ${BIND_HOST} --port ${BIND_PORT} --workers 2
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

echo "[BRIDGE] Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "[BRIDGE] Enabling and starting ${SERVICE_NAME}..."
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

sleep 2

echo ""
echo "======================================================"
echo "  Semantic Cache Bridge API deployed successfully!"
echo "======================================================"
echo "  Service name : ${SERVICE_NAME}"
echo "  Listening on : http://${BIND_HOST}:${BIND_PORT}"
echo "  Health check : http://<your-ec2-ip>:${BIND_PORT}/health"
echo "  API docs     : http://<your-ec2-ip>:${BIND_PORT}/docs"
echo ""
echo "  Status  : sudo systemctl status ${SERVICE_NAME}"
echo "  Logs    : journalctl -u ${SERVICE_NAME} -f"
echo "  Stop    : sudo systemctl stop ${SERVICE_NAME}"
echo "======================================================"
echo ""
echo "  Quick test (replace <secret> with your BRIDGE_API_KEY):"
echo "  curl -s -X POST http://localhost:${BIND_PORT}/v1/cache/query \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -H 'X-API-Key: <secret>' \\"
echo "    -d '{\"prompt\": \"What is machine learning?\"}' | python3 -m json.tool"
echo ""




