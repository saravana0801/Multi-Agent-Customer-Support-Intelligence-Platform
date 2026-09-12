"""
Demonstration and validation runner for CustomerSupportFlow.
Runs 3 representative test cases:
1. Standard Delivery Tracking (Auto-Resolution)
2. High-Value Return ($145.00) (HITL Gate Trigger)
3. Furious Chargeback Complaint (Crisis Escalation)
"""

import json
import time
from src.flow.main_flow import CustomerSupportFlow
from src.flow.state import TicketState
from src.gateway.langfuse_tracker import flush_traces


def run_scenario(title: str, ticket_id: str, query: str):
    print("\n" + "=" * 75)
    print(f"RUNNING: {title}")
    print("=" * 75)
    print(f"Customer Input: \"{query}\"")

    flow = CustomerSupportFlow()
    flow.state.ticket_id = ticket_id
    flow.state.raw_query = query

    start_time = time.perf_counter()
    flow.kickoff()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    print("\n--- Flow Execution Trace ---")
    for step in flow.state.execution_steps:
        print(f"  -> {step}")

    print("\n--- Flow Pydantic State Outputs ---")
    print(f"  • Category:          {flow.state.category} (Confidence: {flow.state.confidence_score})")
    print(f"  • Priority:          {flow.state.priority}")
    print(f"  • Sentiment:         {flow.state.sentiment}")
    print(f"  • Escalation Risk:   {flow.state.escalation_risk}")
    print(f"  • Extracted Entities:{flow.state.entities}")
    print(f"  • Requires HITL:     {flow.state.requires_hitl} (Reason: {flow.state.hitl_reason})")
    print(f"  • Auto-Resolved:     {flow.state.auto_resolved}")
    print(f"  • Escalated:         {flow.state.escalated}")
    print(f"  • Resolution Source: {flow.state.resolution_source}")
    print(f"  • Total Flow Time:   {elapsed_ms:.2f} ms")
    if flow.state.trace_id:
        print(f"  • Langfuse Trace ID: {flow.state.trace_id}")
    if flow.state.trace_url:
        print(f"  • Langfuse Trace:    {flow.state.trace_url}")

    if flow.state.retrieved_evidence:
        print(f"\n--- Retrieved Top Evidence ({len(flow.state.retrieved_evidence)} sources) ---")
        for ev in flow.state.retrieved_evidence[:2]:
            print(f"  [{ev['type']} | {ev['id']}] Similarity: {ev['similarity']}")

    print("\n--- Final Generated Customer Response ---")
    print(flow.state.final_response)


def main():
    # Scenario 1: Delivery Tracking (Happy Path Auto-Resolution)
    run_scenario(
        title="Scenario 1: Delivery Tracking (Happy Path Auto-Resolution)",
        ticket_id="TCK-1001",
        query="Hi, where is my order #ORD5614226? I want to track my package."
    )

    # Scenario 2: High-Value Return ($145.00) (HITL Gate Trigger)
    run_scenario(
        title="Scenario 2: High-Value Return ($145.00) (Financial HITL Gate)",
        ticket_id="TCK-1002",
        query="I received the wrong item in order #ORD9503123. The jacket cost me $145.00 and I want to return it for a refund."
    )

    # Scenario 3: Furious Customer / Double Charge (Crisis Fast-Track Escalation)
    run_scenario(
        title="Scenario 3: Furious Customer / Threat (Crisis Fast-Track Escalation)",
        ticket_id="TCK-1003",
        query="This is totally unacceptable! You charged me twice for my order and your support has been useless! Refund my money immediately or I am reporting fraud and filing a chargeback with my bank!"
    )

    # Flush all traces to Langfuse Cloud
    print("\n--- Flushing telemetry batches to Langfuse Cloud ---")
    flush_traces()
    print("✓ All traces successfully submitted.")


if __name__ == "__main__":
    main()
