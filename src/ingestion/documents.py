from langchain_classic.docstore.document import Document as LangchainDocument
from tqdm import tqdm

def build_raw_knowledge_base(dataset):
    return [
        LangchainDocument(
            page_content=doc["text"],
            metadata={"source": doc["source"]},
        )
        for doc in tqdm(dataset)
    ]
