import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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
