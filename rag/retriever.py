import os
import streamlit as st

PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vectorstore"
)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def get_embeddings():
    """
    Lazy loads and caches the HuggingFace embeddings model.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

@st.cache_resource
def get_vectorstore():
    """
    Lazy loads and builds the Chroma vector store in-memory from the source files.
    This bypasses the Windows/Python 3.13 SQLite Rust panic and file-locking issues.
    """
    from langchain_community.vectorstores import Chroma
    from ingestion.load_manuals import load_manuals
    from ingestion.load_logs import load_logs
    from ingestion.load_schedule import load_schedule
    from ingestion.load_troubleshooting import load_troubleshooting

    # Load all documents from files
    manual_docs = load_manuals()
    log_docs = load_logs()
    schedule_docs = load_schedule()
    troubleshooting_docs = load_troubleshooting()

    all_docs = manual_docs + log_docs + schedule_docs + troubleshooting_docs
    embeddings = get_embeddings()

    # Build the Chroma database in-memory
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings
    )
    return vectorstore

def get_retriever():
    """
    Returns the vectorstore formatted as a retriever.
    """
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": 4}) if vs else None

def reload_vectorstore():
    """
    Clears the cached vectorstore and embeddings. They will be rebuilt
    automatically on the next retrieval request.
    """
    get_vectorstore.clear()
    get_embeddings.clear()

# Dynamic module-level attribute lookup for backward compatibility
def __getattr__(name):
    if name == "vectorstore":
        return get_vectorstore()
    if name == "embeddings":
        return get_embeddings()
    if name == "retriever":
        return get_retriever()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
