import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import get_vectorstore


def schedule_search(query: str) -> list:
    """
    Searches ChromaDB for maintenance schedule documents.
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return []

    return vectorstore.similarity_search(
        query,
        k=4,
        filter={"type": {"$eq": "maintenance_schedule"}}
    )
