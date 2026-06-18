#!/bin/bash
set -e

echo "========================================="
echo "FLUXO Commuter APK Builder"
echo "========================================="

cd "$(dirname "$0")"

echo ""
echo "Step 1: Installing dependencies..."
npm install

echo ""
echo "Step 2: Building web app..."
npm run build

echo ""
echo "Step 3: Initializing Capacitor..."
npx cap init "FLUXO Commuter" "com.fluxo.commuter" --web-dir dist 2>/dev/null || true

echo ""
echo "Step 4: Adding Android platform..."
npx cap add android 2>/dev/null || true

echo ""
echo "Step 5: Syncing to Android..."
npx cap sync android

echo ""
echo "Step 6: Building APK..."
cd android
./gradlew assembleDebug

APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
if [ -f "$APK_PATH" ]; then
    cp "$APK_PATH" "../fluxo-commuter.apk"
    echo ""
    echo "========================================="
    echo "APK Built Successfully!"
    echo "========================================="
    echo "APK location: commuter-app/fluxo-commuter.apk"
    echo "Size: $(du -h ../fluxo-commuter.apk | cut -f1)"
    echo ""
    echo "To install on phone:"
    echo "  adb install fluxo-commuter.apk"
    echo "  OR transfer the APK to your phone and install"
else
    echo "ERROR: APK not found at $APK_PATH"
    echo "Make sure Android SDK is installed"
fi
