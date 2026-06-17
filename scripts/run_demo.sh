#!/bin/bash
set -e

echo "Starting FLUXO demo stack..."

# Start infrastructure
docker-compose up -d redis postgres
sleep 3

# Start backend
echo "Starting backend..."
uvicorn core.api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start dashboard
echo "Starting dashboard..."
cd dashboard && npm run dev &
DASHBOARD_PID=$!

echo ""
echo "FLUXO is running!"
echo "  Backend:   http://localhost:8000"
echo "  Dashboard: http://localhost:5173"
echo "  API docs:  http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

trap "kill $BACKEND_PID $DASHBOARD_PID; docker-compose down" EXIT
wait
