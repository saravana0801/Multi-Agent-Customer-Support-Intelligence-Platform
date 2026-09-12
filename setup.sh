#!/usr/bin/env bash
# ==============================================================================
# Multi-Agent Customer Support Intelligence Platform: Setup & Bootstrap Script
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "🛍️  Setting up Multi-Agent Customer Support Intelligence Platform"
echo "========================================================================"

# 1. Detect Python (Prefer Python 3.11, then 3.10, then python3)
PYTHON_BIN=""
if command -v /opt/homebrew/bin/python3.11 &>/dev/null; then
    PYTHON_BIN="/opt/homebrew/bin/python3.11"
elif command -v python3.11 &>/dev/null; then
    PYTHON_BIN="python3.11"
elif command -v python3.10 &>/dev/null; then
    PYTHON_BIN="python3.10"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "❌ Error: Python 3 (>= 3.10) is required but not found."
    exit 1
fi

echo "✓ Using Python: $($PYTHON_BIN --version) ($PYTHON_BIN)"

# 2. Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv/..."
    $PYTHON_BIN -m venv .venv
fi
source .venv/bin/activate
echo "✓ Virtual environment activated (.venv)"

# 3. Install Dependencies
echo "Installing and updating dependencies from requirements.txt..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✓ All dependencies installed successfully."

# 4. Environment Variables Template
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi
export CREWAI_TELEMETRY_OPT_OUT=true
export CREWAI_DISABLE_TELEMETRY=true
export PYTHONPATH=.

# 5. Create storage directories
mkdir -p logs models chroma_db data
mkdir -p "$HOME/Library/Application Support/AgenticAi/memory" 2>/dev/null || true

# 6. Build Synthetic OMS Data (if not already built)
if [ ! -f "data/mock_oms_orders.json" ]; then
    echo "Generating synthetic PostgreSQL OMS dataset..."
    python3 src/database/generate_oms_data.py
    echo "✓ OMS database generated."
fi

# 7. Build ChromaDB Vector Store (if not already built)
if [ ! -f "chroma_db/chroma.sqlite3" ]; then
    echo "Indexing ChromaDB Knowledge Base (150 FAQs + 696 Golden Tickets)..."
    python3 src/rag/build_chroma_knowledge_base.py
    echo "✓ ChromaDB collections built."
fi

# 8. Train ML Baseline Models (if not already trained)
if [ ! -f "models/category_model.joblib" ]; then
    echo "Training ML Baseline Models on 10,000 Support Tickets..."
    python3 src/models/train_baseline_models.py
    echo "✓ ML Baseline models trained."
fi

echo ""
echo "========================================================================"
echo "🎉 Setup Complete! Everything is ready to launch."
echo "========================================================================"
echo ""
echo "To start the application:"
echo "  1. Quick Launch (Both API and UI):"
echo "     ./start.sh"
echo ""
echo "  2. Or run manually in separate terminals:"
echo "     Terminal 1 (FastAPI):  source .venv/bin/activate && uvicorn src.api.server:app --port 8000"
echo "     Terminal 2 (Streamlit): source .venv/bin/activate && streamlit run src/ui/app.py --server.port 8501"
echo ""
echo "Access Points:"
echo "  • Streamlit Web UI: http://127.0.0.1:8501"
echo "  • FastAPI Swagger:  http://127.0.0.1:8000/docs"
echo "========================================================================"
