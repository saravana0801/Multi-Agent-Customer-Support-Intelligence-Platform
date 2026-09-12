"""
Build and persist ChromaDB collections:
1. faq_collection (150 canonical FAQ policy entries)
2. golden_resolutions_collection (696 verified CSAT>=4 auto-resolved support tickets)
"""

import csv
import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_db"
FAQ_FILE = "data/faq_knowledge_base_150.csv"
TICKETS_FILE = "data/support_tickets_10k.csv"


def build_knowledge_base():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    print(f"Connecting to ChromaDB persistent storage at '{CHROMA_PATH}'...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    # ==================== 1. Ingest FAQs ====================
    print("\n--- 1. Ingesting FAQs ---")
    faq_collection = client.get_or_create_collection(
        name="faq_collection",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    faq_ids = []
    faq_docs = []
    faq_metas = []

    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row["faq_id"].strip()
            cat = row["category"].strip()
            q = row["question"].strip()
            a = row["answer"].strip()

            doc_text = f"Category: {cat}\nQuestion: {q}\nOfficial Answer: {a}"
            faq_ids.append(fid)
            faq_docs.append(doc_text)
            faq_metas.append({
                "faq_id": fid,
                "category": cat,
                "question": q,
                "type": "faq"
            })

    # Upsert into collection
    faq_collection.upsert(
        ids=faq_ids,
        documents=faq_docs,
        metadatas=faq_metas
    )
    print(f"Successfully indexed {len(faq_ids)} FAQs into 'faq_collection'.")

    # ==================== 2. Ingest Golden Historical Tickets ====================
    print("\n--- 2. Ingesting Golden Historical Resolutions (CSAT >= 4 & Auto-resolved) ---")
    golden_collection = client.get_or_create_collection(
        name="golden_resolutions_collection",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    tkt_ids = []
    tkt_docs = []
    tkt_metas = []

    with open(TICKETS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csat = int(row.get("customer_satisfaction_score", 0))
            auto_res = row.get("auto_resolved", "").strip().lower()

            # Filter for golden records
            if csat >= 4 and auto_res == "yes":
                tid = row["ticket_id"].strip()
                cat = row["ticket_category"].strip()
                p_name = row.get("product_name", "Unknown").strip()
                p_seg = row.get("product_segment", "General").strip()
                text = row.get("ticket_text", "").strip()
                res = row.get("resolution_text", "").strip()

                doc_text = (
                    f"Category: {cat} | Product: {p_name} ({p_seg})\n"
                    f"Customer Issue: {text}\n"
                    f"Proven Resolution: {res}"
                )

                tkt_ids.append(tid)
                tkt_docs.append(doc_text)
                tkt_metas.append({
                    "ticket_id": tid,
                    "category": cat,
                    "product_name": p_name,
                    "product_segment": p_seg,
                    "csat": csat,
                    "type": "golden_ticket"
                })

    # Batch upsert (batches of 200)
    batch_size = 200
    for i in range(0, len(tkt_ids), batch_size):
        end_idx = min(i + batch_size, len(tkt_ids))
        golden_collection.upsert(
            ids=tkt_ids[i:end_idx],
            documents=tkt_docs[i:end_idx],
            metadatas=tkt_metas[i:end_idx]
        )
    print(f"Successfully indexed {len(tkt_ids)} Golden Tickets into 'golden_resolutions_collection'.")

    # ==================== Verify Counts ====================
    print("\n--- Verification ---")
    print(f"faq_collection count:                 {faq_collection.count()}")
    print(f"golden_resolutions_collection count: {golden_collection.count()}")
    print("Knowledge base initialization complete!")


if __name__ == "__main__":
    build_knowledge_base()
