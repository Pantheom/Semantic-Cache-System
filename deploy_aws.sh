#!/usr/bin/env bash
# ==============================================================================
# AXIOM Semantic Cache — AWS EC2 One-Click Deployment & Restart Script
# ==============================================================================
# Usage on AWS EC2 (Ubuntu/Linux):
#   chmod +x deploy_aws.sh
#   ./deploy_aws.sh
# ==============================================================================

set -e # Exit immediately if any command fails

echo "========================================================"
echo "🚀 Starting AXIOM Hybrid Cache Deployment on AWS EC2..."
echo "========================================================"

# 1. Create logs directory if it doesn't exist
mkdir -p logs

# 2. Stop any running uvicorn processes for our microservices
echo "🛑 Stopping existing Uvicorn server processes..."
pkill -f "uvicorn query_classifier:app" || true
pkill -f "uvicorn main:app" || true
# Optionally kill anything occupying ports 8000 and 8001 directly
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8001/tcp 2>/dev/null || true
sleep 2

# 3. Pull latest code from Git repository
echo "📥 Pulling latest updates from Git repository..."
git pull origin || git pull || echo "⚠️ Git pull skipped or not in tracking branch."

# 4. Activate Virtual Environment & Check Dependencies
if [ -d ".venv" ]; then
    echo "🐍 Activating existing virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "🐍 Activating existing virtual environment (venv)..."
    source venv/bin/activate
else
    echo "⚠️ No virtual environment folder found (.venv or venv). Using system/global python."
fi

echo "📦 Verifying dependencies..."
pip install -r requirements.txt --quiet

# 5. Start the Query Classifier Service (Port 8001)
echo "⚡ Starting Query Classifier Service on Port 8001..."
nohup uvicorn query_classifier:app --host 0.0.0.0 --port 8001 > logs/classifier.log 2>&1 &
CLASSIFIER_PID=$!
echo "   ↳ Classifier started with PID: $CLASSIFIER_PID (Log: logs/classifier.log)"

# Give the classifier model 3 seconds to begin loading into memory
sleep 3

# 6. Start the Main Hybrid Cache API (Port 8000)
echo "⚡ Starting Main Hybrid Cache API on Port 8000..."
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > logs/main_api.log 2>&1 &
MAIN_PID=$!
echo "   ↳ Main API started with PID: $MAIN_PID (Log: logs/main_api.log)"

echo "========================================================"
echo "✅ Deployment Successful! Both servers are running in background."
echo "========================================================"
echo "💡 To check live status or watch logs on AWS:"
echo "   tail -f logs/main_api.log"
echo "   tail -f logs/classifier.log"
echo "========================================================"
