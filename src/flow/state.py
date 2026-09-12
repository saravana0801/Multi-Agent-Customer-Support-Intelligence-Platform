"""
Strongly typed Pydantic State for Customer Support CrewAI Flow.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TicketCategory(str, Enum):
    REFUND_RETURN = "Refund & Return"
    DELIVERY_ISSUE = "Delivery Issue"
    PRODUCT_ISSUE = "Product Issue"
    PAYMENT_ISSUE = "Payment Issue"
    ACCOUNT_LOGIN = "Account & Login"
    SELLER_PRODUCT = "Seller & Product Listing"
    APP_WEBSITE = "App & Website Issue"
    GENERAL = "General"


class TicketPriority(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class TicketState(BaseModel):
    # 1. Inputs
    ticket_id: str = Field(default="", description="Unique support ticket identifier")
    customer_id: str = Field(default="CUST-1001", description="Customer identifier")
    raw_query: str = Field(default="", description="Raw customer message")
    cleaned_query: str = Field(default="", description="Sanitized and normalized text")

    # 2. Security & Guardrails
    security_flagged: bool = Field(default=False, description="Whether prompt injection or malicious attack was detected")
    security_reason: Optional[str] = Field(default=None)
    out_of_domain: bool = Field(default=False, description="Whether query is outside e-commerce customer support domain")
    domain_rejection_reason: Optional[str] = Field(default=None)
    pii_redacted: List[str] = Field(default_factory=list)

    # 3. ML Triage Outputs (Sub-3ms)
    category: str = Field(default=TicketCategory.GENERAL.value)
    priority: str = Field(default=TicketPriority.LOW.value)
    sentiment: str = Field(default="Neutral")
    confidence_score: float = Field(default=0.0)
    escalation_risk: bool = Field(default=False)
    entities: Dict[str, Any] = Field(default_factory=dict)
    order_id: Optional[str] = Field(default=None)
    amount: float = Field(default=0.0)
    tracking_number: Optional[str] = Field(default=None)

    # 4. Retrieval Context (ChromaDB)
    retrieved_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_context_str: str = Field(default="")

    # 5. Operational OMS Data (PostgreSQL / Mock)
    order_details: Optional[Dict[str, Any]] = Field(default=None)

    # 6. Drafting, Escalation, & HITL Lifecycle
    draft_response: str = Field(default="")
    requires_hitl: bool = Field(default=False)
    hitl_reason: Optional[str] = Field(default=None)
    hitl_decision: Optional[str] = Field(default=None)
    final_response: str = Field(default="")
    escalated: bool = Field(default=False)
    auto_resolved: bool = Field(default=False)
    resolution_source: str = Field(default="")

    # 7. Audit Trail & Tracing
    execution_steps: List[str] = Field(default_factory=list)
    flow_latency_ms: float = Field(default=0.0)
    trace_id: Optional[str] = Field(default=None, description="Langfuse trace identifier")
    trace_url: Optional[str] = Field(default=None, description="Langfuse cloud trace dashboard URL")
