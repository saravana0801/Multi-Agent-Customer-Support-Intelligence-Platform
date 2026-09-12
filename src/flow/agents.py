"""
Response Synthesis Engine:
Orchestrates policy-grounded customer response drafting via the LiteLLM Gateway
with automatic fallback to verified e-commerce resolution templates.
"""

import os
from typing import Optional, Dict, Any, List
from src.gateway.llm_gateway import gateway
from src.gateway.langfuse_tracker import langfuse_observe


@langfuse_observe(name="synthesize_customer_response", as_type="agent")
def synthesize_customer_response(
    ticket_id: str,
    raw_query: str,
    category: str,
    retrieved_evidence: List[Dict[str, Any]],
    order_details: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Synthesizes an empathetic, policy-grounded customer response.
    Routes to LiteLLM Gateway (Groq -> NVIDIA NIM -> OpenAI) if API keys are configured,
    otherwise executes zero-cost grounded template synthesis.
    """
    has_live_key = bool(
        os.environ.get("GROQ_API_KEY") or
        os.environ.get("NVIDIA_API_KEY") or
        os.environ.get("OPENAI_API_KEY") or
        os.environ.get("GEMINI_API_KEY")
    )

    if has_live_key:
        prompt = (
            f"Customer Query: {raw_query}\n"
            f"Category: {category}\n"
            f"Order Details: {order_details}\n"
            f"Retrieved Policy & Precedents Context:\n{retrieved_evidence}\n\n"
            "Please draft a professional, polite, and concise response to the customer."
        )
        try:
            gw_res = gateway.generate(
                prompt=prompt,
                context=str(retrieved_evidence),
                session_id=ticket_id,
                generation_name=f"agent-draft-{category.lower().replace(' ', '-')}",
                tags=["agent-draft", category],
            )
            if gw_res and gw_res.get("response"):
                return gw_res["response"]
        except Exception as e:
            print(f"[ResponseSynthesis] Gateway call error, using grounded template: {e}")

    # Deterministic Grounded Policy Synthesis
    top_item = retrieved_evidence[0] if retrieved_evidence else None
    source_citation = f"[{top_item['type']}: {top_item['id']}]" if top_item else "[Official Store Policy]"

    if category == "Delivery Issue":
        if order_details and order_details.get("tracking_number"):
            carrier = order_details.get("carrier", "our courier partner")
            trk = order_details.get("tracking_number")
            status = order_details.get("status", "In Transit")
            return (
                f"Hello! Thank you for reaching out regarding your order ({order_details.get('order_id', 'referenced order')}).\n\n"
                f"According to our live tracking records, your package is currently **{status}** with {carrier} "
                f"(Tracking Number: `{trk}`). You can track real-time transit updates via 'My Orders > Track Package' {source_citation}.\n\n"
                f"Please let us know if you need any further assistance!"
            )
        return (
            f"Hello! Thank you for contacting customer support.\n\n"
            f"To check live delivery status, please head over to 'My Orders > Select Order > Track Package' for real-time tracking {source_citation}. "
            f"If your order shows delivered but has not arrived, please let us know within 48 hours so we can initiate an immediate carrier investigation."
        )

    elif category == "Refund & Return":
        return (
            f"Hello! Thank you for getting in touch regarding your return request.\n\n"
            f"Per our standard store return guidelines {source_citation}, returns are accepted within our return window. "
            f"We have initiated the return authorization for your item. Once our courier picks up the parcel and it passes quality inspection, "
            f"your refund will disburse to your original payment method within 2-5 business days."
        )

    elif category == "Product Issue":
        return (
            f"Hello! We are truly sorry to hear that your product encountered an issue.\n\n"
            f"According to our product quality policy {source_citation}, defective items are eligible for an immediate replacement or full refund. "
            f"Our team has scheduled a reverse pickup for inspection, and a replacement order has been flagged for dispatch."
        )

    else:
        return (
            f"Hello! Thank you for contacting customer support regarding {category}.\n\n"
            f"We have reviewed your request per store guidelines {source_citation}. "
            f"Our team is actively processing your inquiry and will ensure this is fully resolved for you."
        )


# Backward-compatibility stubs
def create_response_agent(llm_model: Optional[str] = None):
    return None

def create_escalation_agent(llm_model: Optional[str] = None):
    return None
