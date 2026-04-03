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
    print("📂 Loading manuals...")
    manual_docs = load_manuals()
    print(f"   → {len(manual_docs)} manual chunks loaded.")

    print("📋 Loading maintenance logs...")
    log_docs = load_logs()
    print(f"   → {len(log_docs)} log entries loaded.")

    print("📅 Loading maintenance schedule...")
    schedule_docs = load_schedule()
    print(f"   → {len(schedule_docs)} schedule entries loaded.")

    print("🔧 Loading troubleshooting guide...")
    troubleshooting_docs = load_troubleshooting()
    print(f"   → {len(troubleshooting_docs)} troubleshooting sections loaded.")

    all_docs = manual_docs + log_docs + schedule_docs + troubleshooting_docs
    print(f"\n📦 Total documents to embed: {len(all_docs)}")

    print("🔄 Initialising embedding model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"💾 Building ChromaDB vector store at: {PERSIST_DIR}")
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    vectorstore.persist()
    print(f"✅ Vector store built with {len(all_docs)} documents.")
    return vectorstore


if __name__ == "__main__":
    build_vector_store()
