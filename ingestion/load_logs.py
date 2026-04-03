import pandas as pd
import os
from langchain_core.documents import Document

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "logs", "maintenance_logs.csv"
)


def load_logs(file_path: str = None) -> list:
    """
    Loads maintenance log CSV and converts each row to a LangChain Document.
    """
    if file_path is None:
        file_path = _DEFAULT_PATH

    df = pd.read_csv(file_path)
    docs = []

    for _, row in df.iterrows():
        content = (
            f"Date: {row['date']}\n"
            f"Machine: {row['machine']}\n"
            f"Issue: {row['issue']}\n"
            f"Action Taken: {row['action_taken']}"
        )
        metadata = {
            "machine": row["machine"],
            "type": "maintenance_log",
            "date": str(row["date"])
        }
        docs.append(Document(page_content=content, metadata=metadata))

    return docs
