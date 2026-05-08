# IMPORTANT: This file uses langchain_community (older API), NOT langchain_chroma.
# This is intentional for compatibility with the ingestion pipeline.

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.load_manuals import load_manuals
from ingestion.load_logs import load_logs
from ingestion.load_schedule import load_schedule
from ingestion.load_troubleshooting import load_troubleshooting

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vectorstore"
)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vector_store():
    """
    Runs all loaders, combines their documents, builds and persists a ChromaDB
    vector store using HuggingFace sentence-transformer embeddings.
    """
    print("[1/6] Loading manuals...")
    manual_docs = load_manuals()
    print(f"   -> {len(manual_docs)} manual chunks loaded.")

    print("[2/6] Loading maintenance logs...")
    log_docs = load_logs()
    print(f"   -> {len(log_docs)} log entries loaded.")

    print("[3/6] Loading maintenance schedule...")
    schedule_docs = load_schedule()
    print(f"   -> {len(schedule_docs)} schedule entries loaded.")

    print("[4/6] Loading troubleshooting guide...")
    troubleshooting_docs = load_troubleshooting()
    print(f"   -> {len(troubleshooting_docs)} troubleshooting sections loaded.")

    all_docs = manual_docs + log_docs + schedule_docs + troubleshooting_docs
    print(f"\n[5/6] Total documents to embed: {len(all_docs)}")

    print("[5/6] Initialising embedding model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Clear old data if the vectorstore already exists (avoids duplicates)
    if os.path.exists(PERSIST_DIR):
        print("[CLEAN] Clearing old vectorstore data...")
        try:
            old_vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            old_collection = old_vs._collection
            old_ids = old_collection.get()["ids"]
            if old_ids:
                # Delete in batches to avoid memory issues
                batch_size = 5000
                for i in range(0, len(old_ids), batch_size):
                    old_collection.delete(ids=old_ids[i:i + batch_size])
                print(f"   -> Cleared {len(old_ids)} old documents.")
        except Exception as e:
            print(f"   -> Warning: could not clear old store: {e}")

    print(f"[6/6] Building ChromaDB vector store at: {PERSIST_DIR}")
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    vectorstore.persist()
    print(f"[DONE] Vector store built with {len(all_docs)} documents.")
    return vectorstore


if __name__ == "__main__":
    build_vector_store()
