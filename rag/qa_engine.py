import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import vectorstore


def answer_question(question: str) -> str:
    """
    Retrieves top-6 docs from the vectorstore and formats them by type.
    Returns a deduplicated string of relevant answers.
    """
    if vectorstore is None:
        return "Vectorstore not initialised. Run ingestion/build_vector_store.py first."

    docs = vectorstore.similarity_search(question, k=6)

    formatted = []
    seen = set()

    for doc in docs:
        doc_type = doc.metadata.get("type", "unknown")
        content = doc.page_content

        # Debug print
        print(f"[metadata] {doc.metadata}")
        print(f"[preview]  {content[:200]}\n")

        if doc_type == "troubleshooting":
            entry = f"Possible Cause: {content}"
        elif doc_type == "maintenance_log":
            entry = f"Log Entry: {content}"
        elif doc_type == "maintenance_schedule":
            entry = f"Scheduled Task: {content}"
        else:
            entry = content

        # Deduplicate
        key = entry.strip()
        if key not in seen:
            seen.add(key)
            formatted.append(entry)

    if formatted:
        return "\n\n".join(formatted)
    return "No relevant information found."


if __name__ == "__main__":
    test_q = "Why is the roller mill showing excessive vibration?"
    print(f"Q: {test_q}\n")
    print(answer_question(test_q))
