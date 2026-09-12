"""
ChromaDB Hybrid Knowledge Retrieval Tool for CrewAI Agents.
Provides semantic retrieval across:
1. Canonical FAQ Store Policies
2. High-CSAT Golden Historical Ticket Resolutions
"""

import os
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.utils import embedding_functions


class ChromaHybridRAGTool:
    def __init__(self, chroma_path: str = "chroma_db"):
        self.chroma_path = chroma_path
        self._init_chroma()

    def _init_chroma(self):
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.faq_collection = self.client.get_collection(
            name="faq_collection",
            embedding_function=self.embed_fn
        )
        self.golden_collection = self.client.get_collection(
            name="golden_resolutions_collection",
            embedding_function=self.embed_fn
        )

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        faq_limit: int = 2,
        precedent_limit: int = 2,
    ) -> Dict[str, Any]:
        """
        Retrieves matching FAQs and historical resolutions for a query.
        Returns both structured records and a pre-formatted context string for LLM prompting.
        """
        results = []

        # 1. Query FAQs
        if faq_limit > 0:
            faq_filter = None
            if category:
                # Map ticket_category to potential FAQ category names
                faq_filter = {"category": category}

            try:
                faq_res = self.faq_collection.query(
                    query_texts=[query],
                    n_results=faq_limit,
                    where=faq_filter if category else None
                )
            except Exception:
                # Fallback without filter if category had zero items in collection
                faq_res = self.faq_collection.query(
                    query_texts=[query],
                    n_results=faq_limit
                )

            if faq_res and faq_res["ids"] and len(faq_res["ids"][0]) > 0:
                for i, fid in enumerate(faq_res["ids"][0]):
                    dist = faq_res["distances"][0][i] if "distances" in faq_res and faq_res["distances"] else 0.5
                    sim = round(1.0 - dist, 4)
                    meta = faq_res["metadatas"][0][i]
                    doc = faq_res["documents"][0][i]
                    results.append({
                        "id": fid,
                        "type": "FAQ Policy",
                        "category": meta.get("category", "General"),
                        "similarity": sim,
                        "content": doc,
                    })

        # 2. Query Golden Historical Resolutions
        if precedent_limit > 0:
            gold_filter = None
            if category:
                gold_filter = {"category": category}

            try:
                gold_res = self.golden_collection.query(
                    query_texts=[query],
                    n_results=precedent_limit,
                    where=gold_filter if category else None
                )
            except Exception:
                gold_res = self.golden_collection.query(
                    query_texts=[query],
                    n_results=precedent_limit
                )

            if gold_res and gold_res["ids"] and len(gold_res["ids"][0]) > 0:
                for i, tid in enumerate(gold_res["ids"][0]):
                    dist = gold_res["distances"][0][i] if "distances" in gold_res and gold_res["distances"] else 0.5
                    sim = round(1.0 - dist, 4)
                    meta = gold_res["metadatas"][0][i]
                    doc = gold_res["documents"][0][i]
                    results.append({
                        "id": tid,
                        "type": "Historical Precedent",
                        "category": meta.get("category", "General"),
                        "product": meta.get("product_name", "General"),
                        "csat": meta.get("csat", 5),
                        "similarity": sim,
                        "content": doc,
                    })

        # Sort by similarity descending
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)

        # 3. Format context string for LLM Agent prompt
        context_blocks = []
        for item in results:
            if item["type"] == "FAQ Policy":
                context_blocks.append(
                    f"--- Official Policy [{item['id']}] (Relevance: {item['similarity']}) ---\n{item['content']}"
                )
            else:
                context_blocks.append(
                    f"--- Verified Past Resolution [{item['id']}] (Product: {item['product']} | CSAT: {item['csat']}/5 | Relevance: {item['similarity']}) ---\n{item['content']}"
                )

        formatted_context = "\n\n".join(context_blocks) if context_blocks else "No matching knowledge base entries found."

        return {
            "query": query,
            "category_filter": category,
            "count": len(results),
            "results": results,
            "formatted_context": formatted_context,
        }

    def add_feedback_precedent(
        self,
        ticket_id: str,
        query: str,
        resolution: str,
        category: str = "General",
        product_name: str = "General",
        product_segment: str = "E-Commerce",
        csat: int = 5,
    ) -> bool:
        """
        Continuous Learning Loop: Ingests verified/approved human supervisor resolutions
        directly back into the golden_resolutions_collection for future RAG retrieval.
        """
        try:
            doc_text = (
                f"Category: {category} | Product: {product_name} ({product_segment})\n"
                f"Customer Issue: {query}\n"
                f"Proven Resolution: {resolution}"
            )
            metadata = {
                "ticket_id": ticket_id,
                "category": category,
                "product_name": product_name,
                "product_segment": product_segment,
                "csat": csat,
                "type": "supervisor_approved_precedent",
            }
            self.golden_collection.upsert(
                ids=[ticket_id],
                documents=[doc_text],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            print(f"[ChromaHybridRAGTool] Warning: Failed to store feedback precedent: {e}")
            return False


# Direct test entrypoint
if __name__ == "__main__":
    tool = ChromaHybridRAGTool()
    test_query = "The lipstick I received stopped working after 5 days, how do I get a refund?"
    output = tool.retrieve(test_query, category="Refund & Return")
    print(f"Query: {test_query}\n")
    print("Formatted Agent Context:\n")
    print(output["formatted_context"])
