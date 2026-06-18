#!/bin/bash
set -e

echo "========================================="
echo "FLUXO Setup Script"
echo "========================================="

cd "$(dirname "$0")"

echo ""
echo "Step 1: Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 2: Dashboard dependencies..."
cd dashboard && npm install && cd ..

echo ""
echo "Step 3: Commuter app dependencies..."
cd commuter-app && npm install && cd ..

echo ""
echo "Step 4: Create directories..."
mkdir -p outputs/clips outputs/results models

echo ""
echo "Step 5: Database setup (if PostgreSQL running)..."
if pg_isready -q 2>/dev/null; then
    echo "PostgreSQL detected, creating database..."
    psql -U postgres -c "CREATE DATABASE fluxo;" 2>/dev/null || echo "Database 'fluxo' may already exist"
    psql -U postgres -c "CREATE USER fluxo WITH PASSWORD 'fluxo_secret';" 2>/dev/null || true
    psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE fluxo TO fluxo;" 2>/dev/null || true
    echo "Database ready"
else
    echo "PostgreSQL not running. Start it or use Docker."
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Quick start:"
echo "  1. Start API:     python3 -m uvicorn core.api.main:app --reload --port 8000"
echo "  2. Start Dashboard: cd dashboard && npm run dev"
echo "  3. Live Camera:   python3 scripts/live_camera.py --source 0 --show --violations"
echo "  4. Docker:        docker-compose up"
echo ""
echo "Phone Camera:"
echo "  1. Install 'IP Webcam' from Play Store"
echo "  2. Start server in the app"
echo "  3. python3 scripts/live_camera.py --source http://<phone_ip>:8080/video --show --violations"
echo ""
echo "Build APK:"
echo "  cd commuter-app && bash build-apk.sh"
