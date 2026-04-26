#!/bin/bash
# 🚀 RAG Demo Deployment Script
# Deploys the Urban Planning RAG demo to GitHub Pages or any static host
# Usage: ./scripts/deploy_demo.sh [target]

set -e

REPO_DIR="/home/fedora/Projects/urban-planning-rag"
DEMO_DIR="$REPO_DIR/demo"
TARGET="${1:-gh-pages}"

echo "🚀 Deploying RAG Demo..."
echo "📂 Source: $DEMO_DIR"

# Check if demo directory exists
if [ ! -d "$DEMO_DIR" ]; then
    echo "❌ Demo directory not found: $DEMO_DIR"
    exit 1
fi

# Get current git branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "📍 Current branch: $BRANCH"

# If deploying to GitHub Pages (gh-pages branch)
if [ "$TARGET" = "gh-pages" ]; then
    echo "📦 Building static site..."
    
    # Create a timestamped deployment record
    echo "<!-- Deployed at $(date -u +'%Y-%m-%d %H:%M:%S UTC') -->" > "$DEMO_DIR/deployment_info.html"
    cat >> "$DEMO_DIR/deployment_info.html" << EOF
<pre>
Git Branch: $BRANCH
Last Commit: $(git log -1 --oneline)
Deployed by: Nine (autonomous agent)
</pre>
EOF

    # Copy to temp location for pages deployment
    DEPLOY_DIR="/tmp/rag-demo-deploy"
    rm -rf "$DEPLOY_DIR"
    cp -r "$DEMO_DIR" "$DEPLOY_DIR"
    
    echo "✅ Demo ready at: $DEPLOY_DIR"
    echo ""
    echo "To deploy to GitHub Pages:"
    echo "  cd $DEPLOY_DIR && git init && git add . && git commit -m 'Demo deploy' && git push"
    echo ""
    echo "Or deploy manually to tanmaychoudhary.com/rag-demo"
fi

# If local testing
if [ "$TARGET" = "local" ]; then
    echo "🧪 Starting local server for testing..."
    cd "$DEMO_DIR"
    python3 -m http.server 8080 &
    PID=$!
    echo "📍 Server running at http://localhost:8080"
    echo "🛑 To stop: kill $PID"
fi

echo ""
echo "✅ Deployment script complete"