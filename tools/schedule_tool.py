import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import vectorstore


def schedule_search(query: str) -> list:
    """
    Searches ChromaDB for maintenance schedule documents.
    """
    if vectorstore is None:
        return []

    return vectorstore.similarity_search(
        query,
        k=4,
        filter={"type": {"$eq": "maintenance_schedule"}}
    )
