from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.docstore.document import Document as LangchainDocument
from transformers import AutoTokenizer


def chunk_documents(
    knowledge_base: List[LangchainDocument],
    chunk_size: int,
    tokenizer_name: str,
) -> List[LangchainDocument]:
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed.extend(text_splitter.split_documents([doc]))

    unique = {}
    result = []
    for doc in docs_processed:
        if doc.page_content not in unique:
            unique[doc.page_content] = True
            result.append(doc)

    return result
