"""
Security Guardrails Facade:
Re-exports clean interfaces backed directly by Guardrails AI Engine (`guardrails_ai_engine.py`).
"""

from typing import Tuple, List
from src.guardrails.guardrails_ai_engine import guardrails_engine


def detect_prompt_injection(query: str) -> Tuple[bool, str, float]:
    """
    Scans customer query for prompt injection using Guardrails AI Validator.
    Returns: (is_injection: bool, threat_type: str, confidence: float)
    """
    is_safe, _, err_msg, _ = guardrails_engine.validate_input(query)
    if not is_safe:
        return True, err_msg or "Prompt injection / jailbreak detected", 0.99
    return False, "Safe", 0.0


def mask_pii(text: str) -> Tuple[str, List[str]]:
    """
    Detects and masks sensitive Personally Identifiable Information (PII)
    using Guardrails AI validators prior to vector DB ingestion or LLM processing.
    """
    _, sanitized, _, redactions = guardrails_engine.validate_input(text)
    return sanitized, redactions


def validate_output_safety(response_text: str) -> Tuple[bool, str]:
    """
    Validates agent output using Guardrails AI before displaying to customer.
    Ensures response does not leak internal instructions or generate fabricated coupons.
    """
    is_safe, validated_text, reason = guardrails_engine.validate_output(response_text)
    if not is_safe:
        return False, f"Output blocked by Guardrails AI: {reason}"

    if len(response_text.strip()) < 15:
        return False, "Response blocked: Output is too brief or incomplete."

    return True, validated_text


def validate_domain_relevance(query: str) -> Tuple[bool, str, str]:
    """
    Validates whether the incoming customer query belongs to the e-commerce support domain.
    Filters out off-topic general knowledge, recipes, coding, trivia, and math.
    Returns: (is_relevant: bool, refusal_response: str, reason: str)
    """
    return guardrails_engine.validate_domain(query)
