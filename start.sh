#!/usr/bin/env bash
# ==============================================================================
# Multi-Agent Customer Support Intelligence Platform: Startup Launcher
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    echo "Virtual environment .venv not found. Running ./setup.sh first..."
    ./setup.sh
fi

source .venv/bin/activate

export CREWAI_TELEMETRY_OPT_OUT=true
export CREWAI_DISABLE_TELEMETRY=true
export POSTHOG_DISABLED=1
export GUARDRAILS_DISABLE_TELEMETRY=true
export PYTHONPATH=.

# Ensure Application Support memory dir exists for CrewAI LanceDB
mkdir -p "$HOME/Library/Application Support/AgenticAi/memory" 2>/dev/null || true

echo "========================================================================"
echo "🚀 Starting Customer Support Intelligence Platform"
echo "========================================================================"

# Trap to kill both background processes on exit (Ctrl+C)
cleanup() {
    echo ""
    echo "Shutting down background services..."
    kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null || true
    wait $FASTAPI_PID $STREAMLIT_PID 2>/dev/null || true
    echo "Services stopped cleanly."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start FastAPI Backend Gateway
echo "Starting FastAPI Gateway on port 8000..."
uvicorn src.api.server:app --host 127.0.0.1 --port 8000 &
FASTAPI_PID=$!

# Wait briefly for FastAPI to bind
sleep 2

# 2. Start Streamlit Interactive UI
echo "Starting Streamlit UI on port 8501..."
streamlit run src/ui/app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!

echo ""
echo "========================================================================"
echo "✅ Both Services are running:"
echo "   • Streamlit Web UI:   http://127.0.0.1:8501"
echo "   • FastAPI Docs:       http://127.0.0.1:8000/docs"
echo "========================================================================"
echo "Press Ctrl+C to terminate both servers."
echo ""

# Wait for both processes
wait $FASTAPI_PID $STREAMLIT_PID
