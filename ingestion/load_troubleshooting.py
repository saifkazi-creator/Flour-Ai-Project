import os
from langchain_core.documents import Document

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "troubleshooting", "troubleshooting_guide.txt"
)


def load_troubleshooting(file_path: str = None) -> list:
    """
    Loads the troubleshooting guide text file and splits it into per-machine
    Documents, one per symptom block.
    """
    if file_path is None:
        file_path = _DEFAULT_PATH

    with open(file_path, "r") as f:
        raw = f.read()

    sections = raw.split("Machine:")
    docs = []

    for section in sections:
        if section.strip() == "":
            continue
        page_content = "Machine:" + section
        machine_name = section.strip().split("\n")[0].strip()
        metadata = {
            "type": "troubleshooting",
            "machine": machine_name
        }
        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs
