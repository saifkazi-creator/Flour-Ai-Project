import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import vectorstore


def manual_search(query: str, source_filter: str = None) -> list:
    """
    Searches ChromaDB for manual documents.

    NOTE: filter_condition is built but intentionally NOT passed to
    similarity_search (per spec). Only the base type=manual filter is used.
    source_filter only constructs the variable for reference.
    """
    if vectorstore is None:
        return []

    # Build filter_condition (not used in similarity_search — intentional)
    if source_filter:
        filter_condition = {
            "$and": [
                {"type": {"$eq": "manual"}},
                {"source": {"$eq": source_filter}}
            ]
        }
    else:
        filter_condition = {"type": {"$eq": "manual"}}  # noqa: F841

    # Always use only the base type filter
    results = vectorstore.similarity_search(
        query, k=6, filter={"type": {"$eq": "manual"}}
    )

    # Post-filter: exclude docs that look like troubleshooting entries
    filtered = [
        doc for doc in results
        if not (
            "possible causes" in doc.page_content.lower()
            and "recommended actions" in doc.page_content.lower()
        )
    ]

    return filtered[:4]
