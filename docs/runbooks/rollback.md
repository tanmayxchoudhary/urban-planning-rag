# Rollback Playbook

**Purpose**: Rollback the Urban Planning RAG deployment to a previous known-good version.
**SLO**: Rollback completes in ≤ 5 minutes.
**Scope**: `urban-rag-embed` service on Lightning AI (`urban-rag-prod-embed` studio).

---

## Overview

This playbook provides procedures for rolling back the deployment when issues are detected.
Rollback targets the Lightning Serve deployment which hosts the embedding service.

### Rollback Targets

| Target | Service | Studio | Rollback Command |
|--------|---------|--------|------------------|
| Embed Service | `urban-rag-embed` | `urban-rag-prod-embed` | `lightning serve rollback` |
| Corpus Version | Qdrant alias | N/A | API call to switch alias |

---

## Pre-Rollback Checks

Before initiating rollback, verify the current state:

```bash
# Check current service health
curl -sf $(lightning serve url urban-rag-embed --studio urban-rag-prod-embed 2>/dev/null)/health

# List available versions
lightning serve list-versions urban-rag-embed --studio urban-rag-prod-embed

# Check current deployed version
lightning serve list 2>/dev/null | grep urban-rag-embed
```

---

## Standard Rollback Procedure

### Step 1: Set Environment Variables

```bash
export LIGHTNING_API_KEY="<your-api-key>"
export SERVICE_NAME="urban-rag-embed"
export STUDIO_NAME="urban-rag-prod-embed"
```

### Step 2: List Available Versions

```bash
echo "=== Available Versions ==="
lightning serve list-versions "$SERVICE_NAME" --studio "$STUDIO_NAME" || true
```

### Step 3: Perform Rollback

```bash
echo "=== Initiating Rollback ==="
lightning serve rollback "$SERVICE_NAME" --studio "$STUDIO_NAME"
```

### Step 4: Wait for Deployment

```bash
echo "=== Waiting for Rollback Deployment (45 seconds) ==="
sleep 45
```

### Step 5: Verify Health

```bash
echo "=== Verifying Service Health ==="
SERVICES_URL=$(lightning serve list 2>/dev/null | grep "$SERVICE_NAME" | awk '{print $4}' || echo "")
if [[ -n "$SERVICES_URL" ]]; then
    HEALTH_URL="${SERVICES_URL}/health"
    echo "Checking: $HEALTH_URL"
    if curl -sf "$HEALTH_URL"; then
        echo ""
        echo "✅ Rollback Successful - Service is healthy"
    else
        echo "❌ Health check failed"
        exit 1
    fi
else
    echo "⚠️ Could not retrieve service URL"
    exit 1
fi
```

---

## Dry-Run Mode

Use `--dry-run` to simulate rollback without making changes:

```bash
#!/bin/bash
# Dry-run rollback validation script
# Run this to verify rollback readiness without making changes

set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-urban-rag-embed}"
STUDIO_NAME="${STUDIO_NAME:-urban-rag-prod-embed}"

echo "=== Rollback Dry-Run Validation ==="
echo "Service: $SERVICE_NAME"
echo "Studio: $STUDIO_NAME"
echo ""

# Check API key
if [[ -z "${LIGHTNING_API_KEY:-}" ]]; then
    echo "❌ LIGHTNING_API_KEY is not set"
    exit 1
fi

# List versions (read-only operation)
echo "[1/4] Listing available versions..."
lightning serve list-versions "$SERVICE_NAME" --studio "$STUDIO_NAME" || true
echo ""

# Check current deployment
echo "[2/4] Checking current deployment status..."
CURRENT_URL=$(lightning serve url "$SERVICE_NAME" --studio "$STUDIO_NAME" 2>/dev/null || echo "")
if [[ -n "$CURRENT_URL" ]]; then
    echo "Current URL: $CURRENT_URL"
    echo "Health check: ${CURRENT_URL}/health"
    if curl -sf "${CURRENT_URL}/health" > /dev/null 2>&1; then
        echo "✅ Current service is healthy"
    else
        echo "⚠️ Current service health check failed (may need rollback)"
    fi
else
    echo "⚠️ Service not currently deployed"
fi
echo ""

# Validate rollback is possible
echo "[3/4] Validating rollback capability..."
# Check if multiple versions exist (prerequisite for rollback)
VERSION_COUNT=$(lightning serve list-versions "$SERVICE_NAME" --studio "$STUDIO_NAME" 2>/dev/null | grep -c "v" || echo "0")
if [[ "$VERSION_COUNT" -lt 2 ]]; then
    echo "⚠️ Only one version available - rollback may not be possible"
else
    echo "✅ Multiple versions available for rollback"
fi
echo ""

echo "[4/4] Dry-run complete - no changes made"
echo ""
echo "To perform actual rollback, run:"
echo "  bash infra/lightning/rollback.sh"
```

**Expected Dry-Run Output:**
```
=== Rollback Dry-Run Validation ===
Service: urban-rag-embed
Studio: urban-rag-prod-embed

[1/4] Listing available versions...
...

[2/4] Checking current deployment status...
Current URL: https://...
Health check: https://.../health
✅ Current service is healthy

[3/4] Validating rollback capability...
✅ Multiple versions available for rollback

[4/4] Dry-run complete - no changes made

To perform actual rollback, run:
  bash infra/lightning/rollback.sh
```

---

## Automated Rollback Script

The project includes an automated rollback script at `infra/lightning/rollback.sh`.

### Usage

```bash
# Full rollback (interactive)
bash infra/lightning/rollback.sh

# With custom service/studio
SERVICE_NAME=urban-rag-embed STUDIO_NAME=urban-rag-prod-embed bash infra/lightning/rollback.sh
```

### Script Features

- ✅ Validates `LIGHTNING_API_KEY` is set
- ✅ Lists available versions before rollback
- ✅ Performs rollback via `lightning serve rollback`
- ✅ Waits 15 seconds for deployment
- ✅ Verifies health endpoint
- ✅ Reports success/failure with clear messaging
- ✅ Completes in < 2 minutes (well under 5-minute SLO)

---

## Corpus Version Rollback (Qdrant)

If a corpus version rollback is needed (separate from service rollback):

```bash
# Get current corpus version
curl -s http://localhost:3100/v1/corpus | jq '.corpus_version'

# List available versions (check Qdrant collections)
curl -s http://localhost:6333/collections | jq '.result.collections[].name'

# Switch alias to previous version
# This is a Qdrant API call - example:
curl -X PUT "http://localhost:6333/collections/pages_alias/points" \
  -H "Content-Type: application/json" \
  -d '{"service": {"retrieval": "<previous-collection-name>"}}'
```

---

## Rollback Verification Checklist

After rollback, verify:

- [ ] `lightning serve list` shows previous version
- [ ] Health endpoint returns 200 OK
- [ ] Service URL is accessible
- [ ] Smoke test passes: `python -m src.eval run --dataset smoke`
- [ ] API `/v1/healthz` returns healthy status

---

## Rollback Time Measurement

To measure actual rollback duration:

```bash
#!/bin/bash
START_TIME=$(date +%s)

echo "=== Timed Rollback ==="
bash infra/lightning/rollback.sh

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "=== Rollback Duration: ${DURATION} seconds ==="

if [[ $DURATION -le 300 ]]; then
    echo "✅ Within 5-minute SLO"
else
    echo "❌ Exceeded 5-minute SLO"
fi
```

---

## Emergency Rollback (Fast Path)

If the standard procedure is too slow or the service is unresponsive:

```bash
#!/bin/bash
# Emergency rollback - skips optional checks for speed
set -euo pipefail

export LIGHTNING_API_KEY="${LIGHTNING_API_KEY:?LIGHTNING_API_KEY is required}"
SERVICE_NAME="${SERVICE_NAME:-urban-rag-embed}"
STUDIO_NAME="${STUDIO_NAME:-urban-rag-prod-embed}"

echo "=== EMERGENCY ROLLBACK ==="

# Immediate rollback
lightning serve rollback "$SERVICE_NAME" --studio "$STUDIO_NAME"

# Reduced wait (30 seconds for startup)
sleep 30

# Quick health check
HEALTH_URL=$(lightning serve url "$SERVICE_NAME" --studio "$STUDIO_NAME" 2>/dev/null)/health
curl -sf "$HEALTH_URL" && echo "✅ Service restored" || echo "⚠️ Check dashboard"
```

---

## Troubleshooting

### Rollback Fails with "No previous version"

```
Error: No previous version available to rollback to
```

**Resolution**: The service only has one deployed version. Deploy a second version first before rollback is possible.

### Health Check Times Out

```bash
# Check Lightning dashboard for deployment status
# URL: https://lightning.ai/studios/<workspace>/<studio>

# Manually verify
lightning serve list
lightning serve logs "$SERVICE_NAME" --studio "$STUDIO_NAME" --last 50
```

### Service URL Not Available

```bash
# Get fresh URL
lightning serve list

# Check service status
lightning serve status "$SERVICE_NAME" --studio "$STUDIO_NAME"
```

---

## Related Documents

- [Deployment Guide](../deployment.md) - Full deployment procedure
- [Lightning Bootstrap](../lightning/bootstrap.md) - Initial setup
- [CI/CD Pipeline](../../.github/workflows/ci-cd.yml) - Automated deployment
