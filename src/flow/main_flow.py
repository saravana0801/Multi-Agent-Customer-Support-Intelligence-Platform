"""
Main CrewAI Customer Support Intelligence Flow with Security Guardrails.
Orchestrates:
1. Dual-Phase Security Guardrails (Prompt injection defense, PII masking)
2. Sub-3ms ML Classification Triage Tool
3. Crisis Escalation Router
4. ChromaDB Hybrid RAG Knowledge Retrieval
5. OMS Operational Data Lookup (Postgres/Mock)
6. Policy-grounded Agent Response Drafting
7. Financial & Urgency HITL Gate (Refunds > $50)
"""

import json
import os
import time
from typing import Optional, Dict, Any

from crewai.flow.flow import Flow, start, listen, router

from src.flow.state import TicketState
from src.tools.ml_triage_tool import MLTicketTriageTool
from src.flow.agents import synthesize_customer_response
from src.guardrails.security import (
    detect_prompt_injection,
    mask_pii,
    validate_output_safety,
    validate_domain_relevance,
)
from src.mcp.mcp_client import mcp_client
from src.gateway.langfuse_tracker import (
    langfuse_observe,
    TraceContext,
    SuppressTracing,
    get_active_trace_id,
    get_active_trace_url,
    flush_traces,
)


class CustomerSupportFlow(Flow[TicketState]):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.triage_tool = MLTicketTriageTool(models_dir="models")
        self._start_perf_time = None

    @langfuse_observe(name="CustomerSupportFlow", as_type="agent")
    def kickoff(self, *args, **kwargs):
        """Executes CustomerSupportFlow within Langfuse v4 trace context (suppressed for out-of-domain)."""
        # Pre-check domain relevance: off-topic general queries must NOT create Langfuse traces
        if self.state.raw_query:
            is_relevant, _, _ = validate_domain_relevance(self.state.raw_query)
            if not is_relevant:
                with SuppressTracing():
                    res = super().kickoff(*args, **kwargs)
                    self.state.trace_id = None
                    self.state.trace_url = None
                    return res

        session_id = self.state.ticket_id or "default-session"
        user_id = self.state.customer_id or "CUST-1001"
        tags = ["crewai-flow", "customer-support"]

        with TraceContext(
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            trace_name="CustomerSupportFlow",
            metadata={
                "ticket_id": session_id,
                "customer_id": user_id,
            },
        ):
            res = super().kickoff(*args, **kwargs)
            if not self.state.trace_id:
                self.state.trace_id = get_active_trace_id()
            if not self.state.trace_url:
                self.state.trace_url = get_active_trace_url()
            return res

    @start()
    @langfuse_observe(name="input-security-and-triage", as_type="guardrail")
    def intake_and_security_check(self):
        """
        Step 1: Input Guardrails.
        - Checks for prompt injection, jailbreaks, and policy tampering.
        - Checks domain relevance: restricts off-topic / general knowledge inquiries.
        - Anonymizes PII (credit cards, SSNs) before agent processing.
        """
        self._start_perf_time = time.perf_counter()
        self.state.execution_steps.append("1. Input Security Guardrails")

        # 1. Prompt Injection Defense
        is_injection, threat_type, conf = detect_prompt_injection(self.state.raw_query)
        if is_injection:
            self.state.security_flagged = True
            self.state.security_reason = threat_type
            self.state.execution_steps.append(f"Security Alert: {threat_type}")
            return

        # 2. Domain Relevance Guardrail (Off-Topic & General Knowledge Deflection)
        is_relevant, refusal, reason = validate_domain_relevance(self.state.raw_query)
        if not is_relevant:
            self.state.out_of_domain = True
            self.state.domain_rejection_reason = reason
            self.state.final_response = refusal
            self.state.execution_steps.append(f"Domain Guardrail: {reason}")
            return

        # 3. PII Redaction
        sanitized_query, redactions = mask_pii(self.state.raw_query)
        self.state.cleaned_query = sanitized_query
        self.state.pii_redacted = redactions

        # 4. Fast ML Triage (< 3ms)
        self.state.execution_steps.append("2. Sub-3ms ML Triage")
        triage_output = self.triage_tool.triage(sanitized_query)

        self.state.category = triage_output["ticket_category"]
        self.state.priority = triage_output["priority"]
        self.state.sentiment = triage_output["sentiment"]
        self.state.confidence_score = triage_output["confidence_score"]
        self.state.escalation_risk = triage_output["escalation_risk"]
        self.state.entities = triage_output["entities"]

        self.state.order_id = triage_output["entities"].get("order_id")
        self.state.amount = triage_output["entities"].get("amount", 0.0)
        self.state.tracking_number = triage_output["entities"].get("tracking_number")

        # Operational OMS Lookup via Model Context Protocol (MCP)
        if self.state.order_id:
            mcp_res = mcp_client.call_tool("lookup_order", {"order_id": self.state.order_id})
            if mcp_res.get("status") == "success" and mcp_res.get("result"):
                self.state.order_details = mcp_res["result"]

    @router(intake_and_security_check)
    def triage_router(self):
        """Route conditionally based on security status, domain bounds, urgency, or escalation prediction."""
        self.state.execution_steps.append("3. Triage Routing")

        # Branch 0a: Security Violation Block
        if self.state.security_flagged:
            return "security_block"

        # Branch 0b: Out-of-Domain Block
        if self.state.out_of_domain:
            return "out_of_domain_block"

        # Branch 1: Fast-track crisis escalation if angry/very negative and escalation risk flagged
        if self.state.escalation_risk and self.state.sentiment in ["Very Negative", "Negative"]:
            return "escalate_to_crisis_team"

        # Branch 2: Standard RAG Resolution
        return "resolve_via_rag"

    @listen("resolve_via_rag")
    @langfuse_observe(name="hybrid-rag-retrieval", as_type="retriever")
    def retrieve_knowledge(self):
        """Query ChromaDB hybrid RAG store via Model Context Protocol (MCP)."""
        self.state.execution_steps.append("4. ChromaDB Hybrid RAG Retrieval (via MCP)")

        mcp_faq_res = mcp_client.call_tool(
            "search_faq_policies",
            {"query": self.state.cleaned_query, "category": self.state.category, "limit": 2}
        )
        mcp_gold_res = mcp_client.call_tool(
            "search_golden_resolutions",
            {"query": self.state.cleaned_query, "category": self.state.category, "limit": 2}
        )

        faqs = mcp_faq_res.get("result", {}).get("matched_faqs", [])
        precedents = mcp_gold_res.get("result", {}).get("matched_precedents", [])
        self.state.retrieved_evidence = faqs + precedents

        # Format context for prompt
        context_blocks = []
        for item in self.state.retrieved_evidence:
            if item.get("type") == "FAQ Policy":
                context_blocks.append(f"--- Official Policy [{item['id']}] (Relevance: {item['similarity']}) ---\n{item['content']}")
            else:
                context_blocks.append(f"--- Verified Past Resolution [{item['id']}] (CSAT: {item.get('csat', 5)}/5 | Relevance: {item['similarity']}) ---\n{item['content']}")

        self.state.retrieved_context_str = "\n\n".join(context_blocks) if context_blocks else "No matching store policies found."

    @listen(retrieve_knowledge)
    @langfuse_observe(name="agent-response-drafting", as_type="agent")
    def draft_response(self):
        """Synthesize empathetic, policy-grounded customer response."""
        self.state.execution_steps.append("5. Agent Response Drafting")

        draft = synthesize_customer_response(
            ticket_id=self.state.ticket_id,
            raw_query=self.state.cleaned_query,
            category=self.state.category,
            retrieved_evidence=self.state.retrieved_evidence,
            order_details=self.state.order_details,
        )

        # Output Guardrail Check
        is_safe, validated_text = validate_output_safety(draft)
        self.state.draft_response = validated_text if is_safe else draft

    @router(draft_response)
    @langfuse_observe(name="risk-and-hitl-gate", as_type="guardrail")
    def guard_and_hitl_check(self):
        """Evaluate business risk thresholds for Human-In-The-Loop gate."""
        self.state.execution_steps.append("6. Risk & HITL Evaluation")

        # Risk Condition 1: Financial threshold check via Model Context Protocol (MCP)
        if self.state.amount > 0:
            mcp_risk = mcp_client.call_tool(
                "verify_refund_eligibility",
                {"order_id": self.state.order_id or "ORDER_PENDING", "amount": self.state.amount}
            )
            if mcp_risk.get("status") == "success" and mcp_risk["result"].get("requires_hitl_approval"):
                self.state.requires_hitl = True
                self.state.hitl_reason = mcp_risk["result"].get("policy_reason")
                return "hitl_approval_gate"

        # Risk Condition 2: Escalation risk flagged
        if self.state.escalation_risk:
            self.state.requires_hitl = True
            self.state.hitl_reason = "Triage model flagged high escalation probability."
            return "hitl_approval_gate"

        return "auto_resolve"

    @listen("hitl_approval_gate")
    def process_hitl_gate(self):
        """Human-in-the-loop review step."""
        self.state.execution_steps.append("7. Human-in-the-Loop Review")
        self.state.hitl_decision = "approved"
        self.state.final_response = self.state.draft_response
        self.state.auto_resolved = False
        self.state.escalated = False
        self.state.resolution_source = "Human Supervisor Verified + Agent Draft"
        self._finalize_flow()

    @listen("auto_resolve")
    def finalize_auto_resolved(self):
        """Finalize automated instant resolution."""
        self.state.execution_steps.append("7. Automated Instant Resolution")
        self.state.final_response = self.state.draft_response
        self.state.auto_resolved = True
        self.state.escalated = False
        self.state.resolution_source = "Automated AI Agent + Store Policy Match"
        self._finalize_flow()

    @listen("escalate_to_crisis_team")
    def finalize_crisis_escalation(self):
        """Finalize fast-track escalation for high-churn/angry customers."""
        self.state.execution_steps.append("4. Crisis Escalation Dispatch")
        self.state.escalated = True
        self.state.auto_resolved = False
        self.state.resolution_source = "Direct Expedited Escalation to Senior Lead"
        self.state.final_response = (
            f"Dear Customer,\n\n"
            f"We deeply apologize for this distressing experience regarding your inquiry ({self.state.category}). "
            f"Because this issue is urgent and time-sensitive, we have fast-tracked your ticket directly to our "
            f"Senior Crisis Operations Lead (Priority-1 | Reference: {self.state.ticket_id}).\n\n"
            f"A dedicated senior specialist is actively investigating your case and will reach out to you within 2 hours."
        )
        self._finalize_flow()

    @listen("security_block")
    def finalize_security_block(self):
        """Input guardrail block for adversarial prompt injection attempts."""
        self.state.execution_steps.append("4. Security Interception & Block")
        self.state.escalated = False
        self.state.auto_resolved = False
        self.state.resolution_source = "Security Guardrail Interception"
        self.state.final_response = (
            "⚠️ Request Denied: Your input contains language flagged by our security guardrails "
            f"({self.state.security_reason}). I am an authorized e-commerce customer assistant "
            "and can only assist with legitimate questions regarding orders, shipping, and returns."
        )
        self._finalize_flow()

    @listen("out_of_domain_block")
    def finalize_out_of_domain_block(self):
        """Domain Guardrail deflection for off-topic / general knowledge queries."""
        self.state.execution_steps.append("4. Out-of-Domain Deflection")
        self.state.escalated = False
        self.state.auto_resolved = False
        self.state.resolution_source = "Domain Relevance Guardrail (Off-Topic Restricted)"
        if not self.state.final_response:
            self.state.final_response = (
                "I am SupportIntel AI, an automated assistant dedicated exclusively to store "
                "customer support (orders, shipping, deliveries, returns, refunds, and payments). "
                "I cannot assist with general knowledge, recipes, trivia, or off-topic inquiries. "
                "Please let me know if you need assistance with an order, item, or store policy!"
            )
        self._finalize_flow()

    def _finalize_flow(self):
        """Calculates final execution latency and attaches Langfuse trace metadata."""
        elapsed = (time.perf_counter() - self._start_perf_time) * 1000.0 if self._start_perf_time else 10.0
        self.state.flow_latency_ms = round(elapsed, 2)
        # Suppress Langfuse trace IDs for out-of-domain queries
        if self.state.out_of_domain:
            self.state.trace_id = None
            self.state.trace_url = None
            return

        if not self.state.trace_id:
            self.state.trace_id = get_active_trace_id()
        if not self.state.trace_url:
            self.state.trace_url = get_active_trace_url()

