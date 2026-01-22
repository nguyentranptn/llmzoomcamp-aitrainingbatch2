from typing import List, Tuple, Optional

from sentence_transformers import CrossEncoder
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.llms import LLM
from langchain_classic.docstore.document import Document as LangchainDocument

from src.rag.prompts import RAG_PROMPT_TEMPLATE


def rerank_documents(
    query: str,
    docs: List[LangchainDocument],
    reranker: CrossEncoder,
    top_k: int,
) -> List[LangchainDocument]:
    """
    Rerank documents using CrossEncoder
    """
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [doc for doc, _ in ranked_docs[:top_k]]


def answer_with_rag(
    question: str,
    llm: LLM,
    knowledge_index: VectorStore,
    reranker: Optional[CrossEncoder] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 7,
) -> Tuple[str, List[LangchainDocument]]:
    """
    Answer a question using RAG with the given knowledge index.

    Steps:
    1. Retrieve documents using vector similarity
    2. (Optional) Rerank documents with CrossEncoder
    3. Build RAG prompt
    4. Generate answer with LLM
    """

    # 1. Retrieve documents
    retrieved_docs = knowledge_index.similarity_search(
        query=question,
        k=num_retrieved_docs,
    )

    # 2. Optional reranking
    if reranker:
        retrieved_docs = rerank_documents(
            query=question,
            docs=retrieved_docs,
            reranker=reranker,
            top_k=num_docs_final,
        )
    else:
        retrieved_docs = retrieved_docs[:num_docs_final]

    # 3. Build context
    context = "\nExtracted documents:\n"
    context += "".join(
        f"Document {i}:::\n{doc.page_content}\n"
        for i, doc in enumerate(retrieved_docs)
    )

    final_prompt = RAG_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )

    # 4. Generate answer
    answer = llm.invoke(final_prompt)

    return answer, retrieved_docs
