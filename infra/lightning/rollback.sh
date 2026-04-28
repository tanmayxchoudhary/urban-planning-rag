#!/bin/bash
# Rollback urban-rag-embed to previous version
# Per PLAN.md §11.9: lightning deploy rollback urban-rag-embed

set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-urban-rag-embed}"
STUDIO_NAME="${STUDIO_NAME:-urban-rag-prod-embed}"

LIGHTNING_API_KEY="${LIGHTNING_API_KEY:-}"

if [[ -z "$LIGHTNING_API_KEY" ]]; then
    echo "ERROR: LIGHTNING_API_KEY environment variable is not set"
    exit 1
fi

echo "=== Rolling back $SERVICE_NAME to previous version ==="

# List available versions first
echo "Available versions:"
lightning serve list-versions "$SERVICE_NAME" --studio "$STUDIO_NAME" || true

# Perform rollback
echo "Performing rollback..."
lightning serve rollback "$SERVICE_NAME" --studio "$STUDIO_NAME"

echo ""
echo "=== Rollback initiated ==="
echo "Verifying service health..."

sleep 15

SERVICES_URL=$(lightning serve list 2>/dev/null | grep "$SERVICE_NAME" | awk '{print $4}' || echo "")
if [[ -n "$SERVICES_URL" ]]; then
    HEALTH_URL="${SERVICES_URL}/health"
    echo "Checking health: $HEALTH_URL"
    curl -sf "$HEALTH_URL" && echo "" && echo "✅ Rollback successful! Service is healthy." || echo "❌ Health check failed after rollback"
else
    echo "⚠️ Could not retrieve service URL. Check Lightning dashboard."
fi
