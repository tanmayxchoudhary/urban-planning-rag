#!/bin/bash
# Bootstrap script for Lightning Studios
# Per PLAN.md §11.3: Provisions the three Studios (dev, batch, prod-embed)

set -euo pipefail

LIGHTNING_USER_ID="${LIGHTNING_USER_ID:-}"
LIGHTNING_API_KEY="${LIGHTNING_API_KEY:-}"

if [[ -z "$LIGHTNING_API_KEY" ]]; then
    echo "ERROR: LIGHTNING_API_KEY environment variable is not set"
    echo "Please set your Lightning AI API key:"
    echo "  export LIGHTNING_API_KEY=<your-key>"
    echo "Get your key at: https://lightning.ai/settings"
    exit 1
fi

echo "=== Urban RAG — Lightning Studio Bootstrap ==="

# Login to Lightning AI
echo "[1/5] Logging into Lightning AI..."
lightning login --key "$LIGHTNING_API_KEY"

# Create the three Studios
echo "[2/5] Provisioning urban-rag-dev Studio (interactive dev, A100 40GB)..."
lightning studio create \
    --name urban-rag-dev \
    --machine A100 \
    --size 40GB \
    --region us-east \
    2>/dev/null || echo "  Studio 'urban-rag-dev' may already exist, skipping..."

echo "[3/5] Provisioning urban-rag-batch Studio (batch indexing, H100 80GB, on-demand)..."
lightning studio create \
    --name urban-rag-batch \
    --machine H100 \
    --size 80GB \
    --region us-east \
    --on-demand \
    2>/dev/null || echo "  Studio 'urban-rag-batch' may already exist, skipping..."

echo "[4/5] Provisioning urban-rag-prod-embed Studio (prod embed, A100 40GB, autoscale)..."
lightning studio create \
    --name urban-rag-prod-embed \
    --machine A100 \
    --size 40GB \
    --region us-east \
    2>/dev/null || echo "  Studio 'urban-rag-prod-embed' may already exist, skipping..."

# Set secrets on the prod studio
echo "[5/5] Setting secrets on urban-rag-prod-embed..."
lightning secrets set \
    --studio urban-rag-prod-embed \
    QDRANT_URL="$QDRANT_URL" \
    QDRANT_API_KEY="${QDRANT_API_KEY:-}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    2>/dev/null || echo "  Warning: Could not set secrets. Set them manually at lightning.ai/settings"

echo ""
echo "=== Bootstrap complete ==="
echo "Studios:"
echo "  urban-rag-dev        : https://lightning.ai/studios/\$LIGHTNING_USER_ID/urban-rag-dev"
echo "  urban-rag-batch      : https://lightning.ai/studios/\$LIGHTNING_USER_ID/urban-rag-batch (on-demand)"
echo "  urban-rag-prod-embed : https://lightning.ai/studios/\$LIGHTNING_USER_ID/urban-rag-prod-embed"
echo ""
echo "Next steps:"
echo "  1. Set QDRANT_URL, QDRANT_API_KEY, HF_TOKEN secrets at lightning.ai/settings"
echo "  2. Run: lightning serve deploy . --config .lightning/embed.yaml"
echo "  3. Verify: curl \$(lightning serve url urban-rag-embed)/health"
