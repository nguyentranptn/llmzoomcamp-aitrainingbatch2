from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.docstore.document import Document as LangchainDocument
from typing import List

def split_documents(
        raw_doc: List[LangchainDocument],
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        add_start_index: bool = True
) -> List[LangchainDocument]:
    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    processed_docs=[]
    for doc in raw_doc:
        processed_docs+=splitter.split_documents([doc])

    return processed_docs