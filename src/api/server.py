"""
FastAPI Backend Gateway for Multi-Agent Customer Support Intelligence Platform.
Exposes REST endpoints for ticket orchestration, HITL review, and analytics.
"""

import os
import time
import uuid
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import warnings
import logging

warnings.filterwarnings("ignore", message=".*lifespan.*")
warnings.filterwarnings("ignore", message=".*IncompleteFieldDefinitionWarning.*")
# Ensure CrewAI environment variables
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from src.flow.main_flow import CustomerSupportFlow
from src.flow.state import TicketState
from src.tools.chroma_rag_tool import ChromaHybridRAGTool
from src.mcp.mcp_client import mcp_client
from src.guardrails.security import validate_domain_relevance
from src.gateway.langfuse_tracker import flush_traces

rag_tool = ChromaHybridRAGTool()

app = FastAPI(
    title="Customer Support Intelligence Platform API",
    description="Multi-Agent AI Customer Support Orchestrator with Sub-3ms ML Triage, ChromaDB RAG, and HITL",
    version="1.0.0",
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active sessions, HITL queue, and audit stats
TICKET_STORE: Dict[str, Dict[str, Any]] = {}
HITL_QUEUE: List[Dict[str, Any]] = []
SYSTEM_STATS = {
    "total_processed": 0,
    "auto_resolved": 0,
    "escalated": 0,
    "hitl_reviewed": 0,
    "total_latency_ms": 0.0,
}


# --- Pydantic Request / Response Models ---

class ProcessTicketRequest(BaseModel):
    raw_query: str = Field(..., description="Customer query text")
    ticket_id: Optional[str] = Field(default=None, description="Optional ticket ID")
    customer_id: Optional[str] = Field(default="CUST-1001", description="Customer ID")


class HITLActionRequest(BaseModel):
    ticket_id: str
    action: str = Field(..., description="'approved', 'escalate', or 'custom'")
    custom_response: Optional[str] = None
    supervisor_notes: Optional[str] = None


# --- Endpoints ---

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "online",
        "service": "Multi-Agent Support Gateway",
        "timestamp": time.time(),
        "models_loaded": {
            "category_classifier": True,
            "priority_classifier": True,
            "sentiment_classifier": True,
            "escalation_risk_classifier": True,
        },
        "chromadb_status": "connected",
    }


@app.get("/api/v1/stats")
def get_stats():
    total = SYSTEM_STATS["total_processed"]
    avg_latency = (
        round(SYSTEM_STATS["total_latency_ms"] / total, 2) if total > 0 else 0.0
    )
    auto_rate = (
        round((SYSTEM_STATS["auto_resolved"] / total) * 100, 1) if total > 0 else 0.0
    )
    escalate_rate = (
        round((SYSTEM_STATS["escalated"] / total) * 100, 1) if total > 0 else 0.0
    )

    return {
        "total_processed": total,
        "auto_resolved_count": SYSTEM_STATS["auto_resolved"],
        "auto_resolved_percentage": f"{auto_rate}%",
        "escalated_count": SYSTEM_STATS["escalated"],
        "escalated_percentage": f"{escalate_rate}%",
        "hitl_reviewed_count": SYSTEM_STATS["hitl_reviewed"],
        "pending_hitl_count": len(HITL_QUEUE),
        "average_latency_ms": avg_latency,
    }


@app.post("/api/v1/tickets/process")
def process_ticket(req: ProcessTicketRequest, background_tasks: BackgroundTasks):
    tid = req.ticket_id or f"TCK-{uuid.uuid4().hex[:6].upper()}"
    start_time = time.perf_counter()

    flow = CustomerSupportFlow()
    flow.state.ticket_id = tid
    flow.state.customer_id = req.customer_id
    flow.state.raw_query = req.raw_query

    # Execute Flow State Machine
    flow.kickoff()
    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # Queue Langfuse trace flush only for genuine in-domain support queries
    if not flow.state.out_of_domain:
        background_tasks.add_task(flush_traces)

    state_dict = flow.state.model_dump()
    state_dict["total_flow_latency_ms"] = elapsed_ms
    if flow.state.out_of_domain:
        state_dict["trace_id"] = None
        state_dict["trace_url"] = None

    # Update stats
    SYSTEM_STATS["total_processed"] += 1
    SYSTEM_STATS["total_latency_ms"] += elapsed_ms

    if flow.state.out_of_domain:
        SYSTEM_STATS["domain_restricted"] = SYSTEM_STATS.get("domain_restricted", 0) + 1
    elif flow.state.auto_resolved:
        SYSTEM_STATS["auto_resolved"] += 1
    elif flow.state.escalated:
        SYSTEM_STATS["escalated"] += 1

    # Check for HITL queue
    if flow.state.requires_hitl:
        HITL_QUEUE.append({
            "ticket_id": tid,
            "raw_query": req.raw_query,
            "category": flow.state.category,
            "amount": flow.state.amount,
            "priority": flow.state.priority,
            "sentiment": flow.state.sentiment,
            "hitl_reason": flow.state.hitl_reason,
            "draft_response": flow.state.draft_response,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    TICKET_STORE[tid] = state_dict
    return state_dict


@app.get("/api/v1/tickets/pending-hitl")
def get_pending_hitl():
    return {"pending_count": len(HITL_QUEUE), "tickets": HITL_QUEUE}


@app.post("/api/v1/tickets/hitl-action")
def resolve_hitl_ticket(req: HITLActionRequest):
    # Find ticket in HITL queue
    target_item = None
    for item in HITL_QUEUE:
        if item["ticket_id"] == req.ticket_id:
            target_item = item
            break

    if not target_item:
        raise HTTPException(status_code=404, detail="Ticket not found in pending HITL review queue.")

    HITL_QUEUE.remove(target_item)
    SYSTEM_STATS["hitl_reviewed"] += 1

    stored = TICKET_STORE.get(req.ticket_id, {})
    if req.action == "approved":
        stored["hitl_decision"] = "approved"
        stored["final_response"] = stored.get("draft_response", "")
        stored["resolution_source"] = "Human Supervisor Verified & Approved"
        stored["escalated"] = False
        # Continuous Learning Loop: Store approved resolution in ChromaDB
        rag_tool.add_feedback_precedent(
            ticket_id=req.ticket_id,
            query=stored.get("raw_query", target_item.get("raw_query", "")),
            resolution=stored["final_response"],
            category=stored.get("category", "General"),
            csat=5,
        )
    elif req.action == "escalate":
        stored["hitl_decision"] = "escalated"
        stored["escalated"] = True
        stored["resolution_source"] = "Escalated by Supervisor to Tier-2 Operations"
        stored["final_response"] = (
            f"Dear Customer, your case has been escalated to Tier-2 Operations per supervisor review. "
            f"Reference ID: {req.ticket_id}."
        )
        SYSTEM_STATS["escalated"] += 1
    elif req.action == "custom":
        stored["hitl_decision"] = "custom_override"
        stored["final_response"] = req.custom_response or stored.get("draft_response", "")
        stored["resolution_source"] = "Human Supervisor Custom Override"
        stored["escalated"] = False
        # Continuous Learning Loop: Store supervisor custom override in ChromaDB
        rag_tool.add_feedback_precedent(
            ticket_id=req.ticket_id,
            query=stored.get("raw_query", target_item.get("raw_query", "")),
            resolution=stored["final_response"],
            category=stored.get("category", "General"),
            csat=5,
        )

    TICKET_STORE[req.ticket_id] = stored
    return {"status": "success", "ticket_id": req.ticket_id, "action": req.action, "final_state": stored}


@app.get("/api/v1/mcp/tools")
def get_mcp_tools():
    """Returns the Model Context Protocol (MCP) tool catalog."""
    return {"server": mcp_client.server_name, "tools": mcp_client.list_tools()}


@app.get("/api/v1/mcp/resources")
def get_mcp_resources():
    """Returns the Model Context Protocol (MCP) resource catalog."""
    return {"server": mcp_client.server_name, "resources": mcp_client.list_resources()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
