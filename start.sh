#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "========================================="
echo "  FLUXO - Adaptive Traffic Intelligence"
echo "========================================="

echo ""
echo "Building frontend..."
cd dashboard && npx vite build --silent 2>/dev/null && cd ..
cd commuter-app && npx vite build --silent 2>/dev/null && cd ..

echo ""
echo "Starting FLUXO on http://localhost:8000"
echo ""
echo "  Dashboard:  http://localhost:8000/dashboard"
echo "  Commuter:   http://localhost:8000/app"
echo "  API Docs:   http://localhost:8000/api"
echo ""

export DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://postgres:postgres@localhost:5432/fluxo}"

python3 -m uvicorn core.api.main:app --host 0.0.0.0 --port 8000 --reload
