#!/bin/bash
# Run batch indexing job on Lightning
# Per PLAN.md §11.6: Triggered via CI when corpus changes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LIGHTNING_API_KEY="${LIGHTNING_API_KEY:-}"
CORPUS_VERSION="${CORPUS_VERSION:-v1.0.0}"
INPUT_PATH="${INPUT_PATH:-s3://urban-rag-corpus/${CORPUS_VERSION}/}"
STUDIO_NAME="${STUDIO_NAME:-urban-rag-batch}"

if [[ -z "$LIGHTNING_API_KEY" ]]; then
    echo "ERROR: LIGHTNING_API_KEY environment variable is not set"
    exit 1
fi

cd "$PROJECT_ROOT"

JOB_NAME="index-${CORPUS_VERSION}-$(date +%Y%m%d-%H%M)"

echo "=== Running batch index job ==="
echo "Job name: $JOB_NAME"
echo "Corpus version: $CORPUS_VERSION"
echo "Input: $INPUT_PATH"
echo "Studio: $STUDIO_NAME"

lightning run job \
    --name "$JOB_NAME" \
    --studio "$STUDIO_NAME" \
    --machine H100 \
    --num_devices 1 \
    --command "cd /root/projects/urban-planning-rag && \
        uv sync && \
        uv run python -m urban_rag.index.batch \
            --corpus-version $CORPUS_VERSION \
            --input $INPUT_PATH \
            --output qdrant://urban-rag-prod" \
    --wait

echo "=== Index job complete ==="
