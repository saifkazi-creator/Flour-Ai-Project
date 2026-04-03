import pandas as pd
import os
from langchain_core.documents import Document

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "schedules", "maintenance_schedule.csv"
)


def load_schedule(file_path: str = None) -> list:
    """
    Loads maintenance schedule CSV and converts each row to a LangChain Document.
    """
    if file_path is None:
        file_path = _DEFAULT_PATH

    df = pd.read_csv(file_path)
    docs = []

    for _, row in df.iterrows():
        content = (
            f"Machine: {row['machine']}\n"
            f"Task: {row['task']}\n"
            f"Frequency: {row['frequency']}"
        )
        metadata = {
            "machine": row["machine"],
            "type": "maintenance_schedule",
            "frequency": row["frequency"]
        }
        docs.append(Document(page_content=content, metadata=metadata))

    return docs
