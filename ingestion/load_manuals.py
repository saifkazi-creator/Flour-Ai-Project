import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

MANUAL_DIRS = [
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "manuals"
    ),
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "uploads"
    ),
]


def load_manuals() -> list:
    """
    Loads and chunks all PDF files from the manuals and uploads directories.
    Tags each chunk with type='manual' metadata.
    """
    # Ensure directories exist
    for d in MANUAL_DIRS:
        os.makedirs(d, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    all_docs = []

    for directory in MANUAL_DIRS:
        if not os.path.exists(directory):
            continue
        for filename in os.listdir(directory):
            if not filename.lower().endswith(".pdf"):
                continue
            full_path = os.path.join(directory, filename)
            try:
                loader = PyPDFLoader(full_path)
                pages = loader.load()
                chunks = splitter.split_documents(pages)
                for chunk in chunks:
                    if len(chunk.page_content.strip()) < 50:
                        continue
                    chunk.metadata["type"] = "manual"
                    chunk.metadata["source"] = os.path.basename(full_path)
                    chunk.metadata["folder"] = directory
                    all_docs.append(chunk)
            except Exception as e:
                print(f"[WARNING] Failed to load {filename}: {e}")

    return all_docs
