# ============================================================
# Urban RAG — Multi-target Dockerfile
# Per PART XI §11.4: three build targets (embed, gateway, web)
# ============================================================

# ------------------------------------------------------------------
# Stage: base — shared dependencies for all Python targets
# ------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps and Python 3.12 from deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install uv for fast package management (bootstrap pip first)
RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --no-cache-dir uv

# Copy project metadata first (layer caching)
COPY pyproject.toml uv.lock* README.md ./

# Sync dependencies (use --frozen for reproducibility)
RUN uv sync --frozen --no-dev

# ------------------------------------------------------------------
# Stage: embed — ColQwen2.5 visual embedding service
# Exposes: /health, /embed, /embed_query on port 3102
# ------------------------------------------------------------------
FROM base AS embed

# Copy the full source tree
COPY src/ src/
COPY PLAN.md ./

# Expose embed service port (per AGENTS.md port range 3100-3199)
EXPOSE 3102

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:3102/health || exit 1

# Run the embed service
CMD ["python", "-m", "urban_rag.embed.serve"]

# ------------------------------------------------------------------
# Stage: gateway — FastAPI API gateway
# Exposes: /v1/* endpoints on port 3100
# ------------------------------------------------------------------
FROM base AS gateway

# Copy the full source tree
COPY src/ src/
COPY PLAN.md ./

# Expose gateway port (per AGENTS.md port range 3100-3199)
EXPOSE 3100

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:3100/v1/healthz || exit 1

# Run the API gateway
CMD ["python", "-m", "uvicorn", "urban_rag.api.main:app", "--host", "0.0.0.0", "--port", "3100"]

# ------------------------------------------------------------------
# Stage: web — Next.js production build
# Exposes: port 3101
# ------------------------------------------------------------------
FROM node:20-alpine AS web-deps

WORKDIR /app

# Copy package files
COPY web/package.json web/package-lock.json* ./

# Install dependencies
RUN npm ci --ci

# ------------------------------------------------------------------
FROM web-deps AS web-builder

WORKDIR /app

# Copy source
COPY web/ ./

# Build Next.js app
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ------------------------------------------------------------------
FROM node:20-alpine AS web

WORKDIR /app

# Copy the entire web app (built during web-builder stage)
COPY web/ ./

# Install production dependencies only
RUN npm ci --ci --omit=dev && npm cache clean --force

# Expose web port (per AGENTS.md port range 3100-3199)
EXPOSE 3101

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3101', (r) => process.exit(r.statusCode === 200 ? 0 : 1)).on('error', () => process.exit(1))" || exit 1

# Start Next.js server directly
CMD ["npm", "start", "--", "-p", "3101"]
