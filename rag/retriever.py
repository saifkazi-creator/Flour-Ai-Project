import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vectorstore"
)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level singletons
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def _load_vectorstore():
    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return None


vectorstore = _load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) if vectorstore else None


def get_retriever():
    return retriever


def reload_vectorstore():
    """
    Re-create the vectorstore and retriever singletons from the persisted
    ChromaDB directory.  Call this after ingestion / rebuild so the running
    app picks up newly added documents.
    """
    global vectorstore, retriever
    vectorstore = _load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) if vectorstore else None

    # Also update the references held by every tool module that already
    # imported `vectorstore` at the top level.
    import tools.manual_tool as _mt
    import tools.log_tool as _lt
    import tools.schedule_tool as _st
    import tools.troubleshooting_tool as _tt
    import rag.qa_engine as _qa

    for mod in (_mt, _lt, _st, _tt, _qa):
        mod.vectorstore = vectorstore

