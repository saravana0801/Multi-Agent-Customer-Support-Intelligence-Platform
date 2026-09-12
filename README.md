# 🛍️ Multi-Agent Customer Support Intelligence Platform

An enterprise-grade, multi-agent customer support intelligence platform built with **CrewAI Flows**, **Sub-3ms Classical ML Triage (0 LLM Tokens)**, **ChromaDB Hybrid Vector Store**, **Guardrails AI (`guardrails-ai`)**, **Official LiteLLM Multi-Model Gateway**, **Model Context Protocol (MCP v1.28)**, and a **Human-in-the-Loop (HITL) Supervisor Inbox with Continuous Learning**.

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    subgraph UI_Layer ["Frontend & Operator Interface (Port 8501)"]
        UI_Chat[💬 Customer Chat Experience]
        UI_Ops[🧠 Agent Ops Decision Trace]
        UI_HITL[🛡️ Supervisor Review Inbox]
        UI_KB[📊 Knowledge Base Explorer]
    end

    subgraph API_Layer ["FastAPI Gateway (Port 8000)"]
        API_Process["POST /api/v1/tickets/process"]
        API_HITL["POST /api/v1/tickets/hitl-action"]
        API_Stats["GET /api/v1/stats"]
    end

    subgraph Guardrails_Layer ["Guardrails AI Engine (guardrails-ai v0.11)"]
        G_In["Input Rails: Prompt Injection Defense & Financial PII Masking"]
        G_Out["Output Rails: Anti-Hallucination & Policy Leak Check"]
    end

    subgraph MCP_Layer ["Model Context Protocol (MCP v1.28)"]
        MCP_Server["FastMCP Server: CustomerSupportMCPServer"]
        MCP_Client["MCP Client Adapter"]
        MCP_Tools["MCP Tools: lookup_order | search_faq | search_precedents | verify_refund"]
        MCP_Res["MCP Resources: support://policies/return-policy | support://metrics/summary"]
        MCP_Server --- MCP_Tools & MCP_Res
        MCP_Client <==>|JSON-RPC| MCP_Server
    end

    subgraph Flow_Layer ["CrewAI Flow State Machine"]
        direction TB
        F_Start([@start: Intake & Guardrails AI])
        F_Triage["Sub-3ms ML Classifier Tool (100% Accuracy)"]
        F_Router{"Triage Router: Urgency & Sentiment"}
        F_RAG["ChromaDB Hybrid RAG (150 FAQs + 696 Golden Tickets)"]
        F_Draft["Empathetic Policy Drafting Agent"]
        F_Risk{"MCP Financial Risk Check (Refund > $50?)"}
        F_HITL["Supervisor Review Gate"]
        F_Auto([Automated Instant Resolution])
        F_Crisis([Crisis Escalation Dispatch])

        F_Start --> F_Triage --> F_Router
        F_Router -->|Routine In-Window| F_RAG --> F_Draft --> F_Risk
        F_Router -->|Angry / Negative / High Escalation Risk| F_Crisis
        F_Risk -->|Amount > $50 / Risk Flag| F_HITL
        F_Risk -->|Standard Policy| F_Auto
    end

    subgraph Gateway_Layer ["LiteLLM Gateway (litellm v1.100)"]
        LLM_Router{"Dynamic Fallback Router"}
        LLM_Groq["Groq: openai/gpt-oss-120b"]
        LLM_NVIDIA["NVIDIA NIM: nvidia/llama-3.1-nemotron-ultra-253b-v1"]
        LLM_Cache["Sub-1ms Semantic Cache"]
        LLM_Offline["Offline Policy Synthesizer"]
        LLM_Router --> LLM_Cache --> LLM_Groq --> LLM_NVIDIA --> LLM_Offline
    end

    subgraph Learning_Layer ["Continuous Learning Loop"]
        HITL_Approve["Supervisor Overrides & Approvals"]
        HITL_Ingest["ChromaDB golden_resolutions_collection Real-Time Ingest"]
        HITL_Approve --> HITL_Ingest
    end

    UI_Chat & UI_Ops & UI_HITL & UI_KB <--> API_Layer
    API_Layer --> G_In --> Flow_Layer
    Flow_Layer <--> MCP_Client
    Flow_Layer --> Gateway_Layer
    Flow_Layer --> G_Out --> API_Layer
    API_HITL --> Learning_Layer
```


---

## ⚡ Quick Start: Setup & Run

### 1. Automated One-Command Bootstrap
Run the setup script to configure the Python 3.11 virtual environment, install dependencies (including `mcp`, `litellm`, and `guardrails-ai`), train ML models, and populate ChromaDB:

```bash
./setup.sh
```

### 2. Launch the Application (API & Web UI)
Launch both the **FastAPI Gateway** (port 8000) and the **Streamlit Web UI** (port 8501) with a single command:

```bash
./start.sh
```

- **Streamlit Interactive UI**: [http://127.0.0.1:8501](http://127.0.0.1:8501)
- **FastAPI Interactive Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔑 LLM Provider Configuration (`.env`)

The platform supports live multi-model execution via **LiteLLM** or zero-token local execution:

```bash
# Copy template
cp .env.example .env
```

Add your API keys to `.env` as desired:
```ini
# Groq Ultra-Fast Inference (Recommended)
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b

# NVIDIA NIM Microservices
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL=nvidia/llama-3.1-nemotron-ultra-253b-v1
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

```

> **Note**: If no API keys are provided, the platform automatically utilizes its **Offline Policy Synthesizer** and local ChromaDB embeddings, ensuring 100% offline functionality at zero token cost.

---

## 📂 Project Structure

```text
AgenticAi/
├── setup.sh                     # 🚀 Automated bootstrap script
├── start.sh                     # ⚡ 1-command startup launcher (FastAPI + Streamlit)
├── setup.py                     # Setuptools package configuration
├── requirements.txt             # Pinned project dependencies
├── mcp_config.json              # Standard Model Context Protocol (MCP) registration
├── .env.example                 # Environment variables template
│
├── data/                        # Datasets & Operational Storage
│   ├── support_tickets_10k.csv  # 10,000 historical support tickets (15 columns)
│   ├── faq_knowledge_base_150.csv # 150 canonical policy FAQ rows (4 categories)
│   ├── mock_oms_orders.json     # 500 synthetic orders for live tool lookup
│   └── init_oms_postgres.sql    # PostgreSQL DDL & batch INSERT seed statements
│
├── models/                      # Serialized ML Inference Artifacts
│   ├── category_model.joblib    # 100.0% Test Accuracy (7 categories)
│   ├── priority_model.joblib    # 91.20% Test Accuracy (High, Medium, Low)
│   ├── escalation_model.joblib  # 82.10% Accuracy (86.79% Precision on escalations)
│   ├── sentiment_model.joblib   # 5-class fine-grained customer sentiment
│   └── baseline_metrics.json    # Complete benchmark evaluation JSON
│
├── chroma_db/                   # Persistent ChromaDB Vector Storage
│   ├── faq_collection           # 150 Canonical Policy FAQ vectors
│   └── golden_resolutions_collection # High-CSAT (>=4) auto-resolved tickets
│

└── src/                         # Core Source Code
    ├── api/
    │   └── server.py            # FastAPI REST Gateway (port :8000)
    │
    ├── ui/
    │   ├── app.py               # Modern Streamlit Web UI (port :8501)
    │   └── style.css            # Custom glassmorphic styling & status badges
    │
    ├── flow/
    │   ├── state.py             # Strongly typed Pydantic TicketState
    │   ├── agents.py            # Streamlined response synthesis engine (LiteLLM Gateway)
    │   ├── main_flow.py         # CrewAI CustomerSupportFlow state machine (Unified MCP Tooling)
    │   └── run_flow_demo.py     # 3-scenario automated demo runner
    │
    ├── tools/
    │   ├── ml_triage_tool.py    # Sub-3ms ML classification & entity extractor
    │   └── chroma_rag_tool.py   # Hybrid vector similarity retrieval & feedback tool
    │
    ├── guardrails/
    │   ├── guardrails_ai_engine.py       # Consolidated Guardrails AI engine & validators
    │   ├── security.py                   # Clean facade re-exporting Guardrails AI engine
    │   └── run_security_and_eval_demo.py # Guardrails verification test suite
    │
    ├── gateway/
    │   └── llm_gateway.py       # LiteLLM multi-provider gateway & semantic cache
    │
    ├── mcp/
    │   ├── support_mcp_server.py # FastMCP Server (OMS lookups, RAG, risk checks)
    │   └── mcp_client.py         # MCP Client Adapter for Flow & Agents
    │
    ├── evaluation/
    │   └── run_eval.py          # Benchmark suite (Accuracy, F1, RAG, SLAs)
    │
    ├── models/
    │   └── train_baseline_models.py # Model training & evaluation pipeline
    │
    └── database/
        └── generate_oms_data.py # Synthetic PostgreSQL OMS generator
```

---

## 🛡️ Enterprise Pillars Deep-Dive

### 1. Guardrails AI (`guardrails-ai v0.11`)
- **Input Rails**:
  - `PromptInjectionValidator`: Detects and neutralizes jailbreaks, developer mode attempts, and policy tampering at **0 token cost**.
  - `PIIMaskingValidator`: Automatically redacts credit cards (`[REDACTED_CREDIT_CARD]`), CVVs (`[REDACTED_CVV]`), and SSNs before data is passed to agents or vector databases.
- **Output Rails**:
  - `OutputSafetyValidator`: Prevents system prompt leaks, meta-prompt exposure, or unauthorized coupon/discount generation.

### 2. Official LiteLLM Multi-Model Gateway (`litellm v1.100`)
- **Dynamic Routing**: Primary model route on **Groq** (`openai/gpt-oss-120b`) with automatic failover to **NVIDIA NIM** (`nvidia/llama-3.1-nemotron-ultra-253b-v1`) and offline policy synthesizer.
- **Semantic Caching**: In-memory cache returns sub-1ms responses for high-frequency inquiries.
- **Spend & Latency Tracking**: Tracks input/output token counts and calculates estimated USD costs per request.

### 3. Model Context Protocol (MCP v1.28)
Built with **FastMCP**, standardizing tools and resources for agents and external MCP clients (Claude Desktop, Antigravity, Cursor):
- **Exposed MCP Tools**:
  - `lookup_order(order_id)`: Operational query to OMS for tracking, carrier, and item manifest.
  - `search_faq_policies(query, category)`: Semantic search over ChromaDB FAQ policies.
  - `search_golden_resolutions(query, category)`: Semantic search over high-CSAT historical precedents.
  - `verify_refund_eligibility(order_id, amount)`: Financial risk check enforcing the $50.00 HITL threshold.
- **Exposed MCP Resources**:
  - `support://policies/return-and-refund`: Canonical return guidelines.
  - `support://metrics/summary`: Live SLA and throughput metrics.
- **MCP Client Adapter**: Discovers tools dynamically and dispatches calls via standard JSON-RPC.

### 4. Continuous Learning Feedback Loop
When human supervisors review tickets in the **Supervisor Review Inbox** (`/api/v1/tickets/hitl-action`):
- Approved drafts or custom supervisor overrides are immediately indexed into ChromaDB's `golden_resolutions_collection` via `add_feedback_precedent()`.
- Newly approved precedents are embedded and retrievable in real-time for subsequent customer queries.

---

## 📊 Benchmark Evaluation Results (`src/evaluation/run_eval.py`)

Evaluation results across the holdout test set (2,000 tickets):

| Evaluation Dimension | Metric | Observed Benchmark | Capstone SLA / Target | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Category Classification** | Accuracy / F1-Score | **100.00% / 100.00%** | > 85.0% | ✅ **PASSED** |
| **Priority Assignment** | Accuracy / F1-Score | **91.20% / 91.08%** | > 85.0% | ✅ **PASSED** |
| **Escalation Prediction** | Precision / Recall | **86.79% / 82.10%** | High Recall on Escalations | ✅ **PASSED** |
| **Sentiment Analysis** | Accuracy / F1-Score | **59.35% / 58.45%** | 5 Fine-Grained Classes | ✅ **PASSED** |
| **Hybrid RAG Context Match** | Category Precision | **100.0%** | Relevant Policy Alignment | ✅ **PASSED** |
| **Hybrid RAG Retrieval** | Top-1 Cosine Similarity | **0.6663** | Cosine Space | ✅ **PASSED** |
| **RAG Retrieval Latency** | Mean Latency | **93.38 ms** | < 150 ms | ✅ **PASSED** |
| **ML Triage Latency (p50)** | Median Inference Time | **1.23 ms** | < 3.0 ms | ✅ **PASSED** |
| **ML Triage Latency (p95)** | 95th Percentile Latency | **1.31 ms** | < 3.0 ms | ✅ **PASSED** |

---

## 🧪 Testing & Verification Commands

```bash
source .venv/bin/activate
export PYTHONPATH=.
export CREWAI_TELEMETRY_OPT_OUT=true
export POSTHOG_DISABLED=1
export GUARDRAILS_DISABLE_TELEMETRY=true

# 1. Run Comprehensive Benchmark Evaluation Suite
python3 src/evaluation/run_eval.py

# 2. Test Model Context Protocol (MCP) Server & Client Adapter
python3 src/mcp/mcp_client.py

# 3. Test Guardrails AI Engine (Jailbreak Defense & PII Redaction)
python3 src/guardrails/guardrails_ai_engine.py

# 4. Test Official LiteLLM Gateway
python3 src/gateway/llm_gateway.py

# 5. Test Full CrewAI Flow across 3 Scenarios
python3 src/flow/run_flow_demo.py

# 6. Test Security Guardrails (Injection Defense & PII Redaction)
python3 src/guardrails/run_security_and_eval_demo.py

```

---

## 📄 License
MIT License. Built for enterprise multi-agent customer support intelligence.
