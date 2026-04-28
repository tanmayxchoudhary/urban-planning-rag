#!/bin/bash
# Deploy urban-rag-embed service via Lightning serve
# Per PLAN.md §11.5: lightning serve deploys urban-rag-embed and it answers /health

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LIGHTNING_API_KEY="${LIGHTNING_API_KEY:-}"
STUDIO_NAME="${STUDIO_NAME:-urban-rag-prod-embed}"
CONFIG_FILE="${PROJECT_ROOT}/.lightning/embed.yaml"

if [[ -z "$LIGHTNING_API_KEY" ]]; then
    echo "ERROR: LIGHTNING_API_KEY environment variable is not set"
    echo "Set it with: export LIGHTNING_API_KEY=<your-key>"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "=== Deploying urban-rag-embed to Lightning ==="
echo "Config: $CONFIG_FILE"
echo "Studio: $STUDIO_NAME"

# Build and push Docker image to GHCR first (per PLAN.md §11.8 CI/CD)
echo "[1/3] Building Docker image..."
docker build \
    --target embed \
    --tag "ghcr.io/$(gh repo view --json owner --jq .owner.login)/urban-rag:embed-$(git rev-parse --short HEAD)" \
    .

echo "[2/3] Pushing to GHCR..."
IMAGE_TAG="ghcr.io/$(gh repo view --json owner --jq .owner.login)/urban-rag:embed-$(git rev-parse --short HEAD)"
docker push "$IMAGE_TAG" || echo "Warning: Could not push image (may need gh login)"

# Deploy via Lightning serve
echo "[3/3] Deploying via Lightning serve..."
lightning serve deploy . \
    --config "$CONFIG_FILE" \
    --studio "$STUDIO_NAME" \
    --verbose

echo ""
echo "=== Deployment initiated ==="
echo "Monitor at: https://lightning.ai/studios"

# Wait for deployment and verify health
echo ""
echo "Waiting for service to become healthy..."
sleep 30

SERVICES_URL=$(lightning serve list 2>/dev/null | grep urban-rag-embed | awk '{print $4}' || echo "")
if [[ -n "$SERVICES_URL" ]]; then
    HEALTH_URL="${SERVICES_URL}/health"
    echo "Checking health: $HEALTH_URL"
    curl -sf "$HEALTH_URL" && echo "" && echo "✅ Service is healthy!" || echo "❌ Health check failed"
else
    echo "⚠️ Could not retrieve service URL. Check Lightning dashboard."
fi
