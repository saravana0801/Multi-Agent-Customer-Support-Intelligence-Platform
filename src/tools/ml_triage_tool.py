"""
High-speed ML Ticket Triage Tool for CrewAI Flow & Agents.
Loads serialized models and provides sub-10ms classification for incoming queries.
"""

import os
import re
import time
from typing import Dict, Any, Optional
import joblib


class MLTicketTriageTool:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self._load_models()

    def _load_models(self):
        self.category_model = joblib.load(os.path.join(self.models_dir, "category_model.joblib"))
        self.priority_model = joblib.load(os.path.join(self.models_dir, "priority_model.joblib"))
        self.sentiment_model = joblib.load(os.path.join(self.models_dir, "sentiment_model.joblib"))
        self.escalation_model = joblib.load(os.path.join(self.models_dir, "escalation_model.joblib"))

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        entities = {}
        # Order reference match: #ORD12345 or ORD-12345 or ORD12345
        order_match = re.search(r"#?(ORD-?\d{5,10})", text, re.IGNORECASE)
        if order_match:
            entities["order_id"] = order_match.group(1).upper()
        else:
            entities["order_id"] = None

        # Monetary amount match: $120 or 120 dollars or USD 120
        amount_match = re.search(r"(\$|usd\s*)(\d+(\.\d{2})?)", text, re.IGNORECASE)
        if amount_match:
            entities["amount"] = float(amount_match.group(2))
        else:
            entities["amount"] = 0.0

        # Tracking match: FDX-123 or UPS-123 or TRACK-123
        track_match = re.search(r"\b(FDX|UPS|DHL|TRK)-\d{6,}\b", text, re.IGNORECASE)
        if track_match:
            entities["tracking_number"] = track_match.group(0).upper()
        else:
            entities["tracking_number"] = None

        return entities

    def triage(self, ticket_text: str) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # Clean text
        clean_text = ticket_text.lower()
        clean_text = re.sub(r"#ord\d+", " ORDER_REF ", clean_text)
        clean_text = re.sub(r"[^\w\s]", " ", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        # Inference
        category = self.category_model.predict([clean_text])[0]
        cat_probs = self.category_model.predict_proba([clean_text])[0]
        confidence = float(max(cat_probs))

        priority = self.priority_model.predict([clean_text])[0]
        sentiment = self.sentiment_model.predict([clean_text])[0]
        escalation = self.escalation_model.predict([clean_text])[0]

        # Entity extraction
        entities = self._extract_entities(ticket_text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "ticket_category": category,
            "confidence_score": round(confidence, 4),
            "priority": priority,
            "sentiment": sentiment,
            "escalation_risk": escalation == "Yes",
            "entities": entities,
            "inference_time_ms": round(elapsed_ms, 2),
        }


# Direct test entrypoint
if __name__ == "__main__":
    triage_tool = MLTicketTriageTool()
    sample_query = "The lipstick I received in order #ORD5614226 stopped working after just 5 days of use. Cost me $45.00!"
    result = triage_tool.triage(sample_query)
    print("Test Query:", sample_query)
    print("Triage Result:", result)
