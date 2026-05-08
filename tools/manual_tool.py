import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import vectorstore


def manual_search(query: str, source_filter: str = None) -> list:
    """
    Searches ChromaDB for manual documents.

    When source_filter is provided (e.g. an uploaded filename), searches that
    specific source first. Falls back to all manuals if no results are found.
    """
    if vectorstore is None:
        return []

    # Build the appropriate filter
    if source_filter:
        filter_condition = {
            "$and": [
                {"type": {"$eq": "manual"}},
                {"source": {"$eq": source_filter}}
            ]
        }
    else:
        filter_condition = {"type": {"$eq": "manual"}}

    # Search with the full filter (including source if provided)
    results = vectorstore.similarity_search(
        query, k=6, filter=filter_condition
    )

    # If source-specific search returned nothing, fall back to all manuals
    if not results and source_filter:
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
