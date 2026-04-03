import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import vectorstore

MACHINES = ["Roller Mill", "Conveyor", "Bucket Elevator", "Purifier"]


def troubleshooting_search(query: str) -> list:
    """
    Searches ChromaDB for troubleshooting documents.
    Optionally filters by detected machine name.
    """
    if vectorstore is None:
        return []

    detected_machine = None
    for machine in MACHINES:
        if machine.lower() in query.lower():
            detected_machine = machine
            break

    if detected_machine:
        return vectorstore.similarity_search(
            query,
            k=4,
            filter={
                "$and": [
                    {"type": {"$eq": "troubleshooting"}},
                    {"machine": {"$eq": detected_machine}}
                ]
            }
        )
    else:
        return vectorstore.similarity_search(
            query,
            k=4,
            filter={"type": {"$eq": "troubleshooting"}}
        )
