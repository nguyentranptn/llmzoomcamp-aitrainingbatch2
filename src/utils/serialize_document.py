import re

def serialize_docs(docs):
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)
