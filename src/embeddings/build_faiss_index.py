from typing import List
import os
from tqdm import tqdm

from langchain_classic.docstore.document import Document as LangchainDocument
from langchain_classic.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from src.chunking.chunk_documents import chunk_documents


def load_or_build_faiss_index(
    langchain_docs: List[LangchainDocument],
    chunk_size: int,
    embedding_model_name: str = "thenlper/gte-small",
    index_root_path: str = "./data/indexes",
    batch_size: int = 32,   
) -> FAISS:
    """
    Load a FAISS index if it exists, otherwise build it with progress bar.
    """

    # 1️⃣ Init embedding model (CPU-safe)
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=False,          # ❗ Windows-safe
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 2️⃣ Build index path
    index_name = (
        f"index_chunk_{chunk_size}_"
        f"embeddings_{embedding_model_name.replace('/', '~')}"
    )
    index_path = os.path.join(index_root_path, index_name)

    # 3️⃣ Load existing index
    if os.path.isdir(index_path):
        print(f"✅ Loading existing FAISS index from {index_path}")
        return FAISS.load_local(
            index_path,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )

    print("🧱 FAISS index not found. Building a new one...")

    # 4️⃣ Split documents
    docs_processed = chunk_documents(
        chunk_size=chunk_size,
        knowledge_base=langchain_docs,
        tokenizer_name=embedding_model_name,
    )

    print(f"📄 Total chunks to embed: {len(docs_processed)}")

    # 5️⃣ Build FAISS index incrementally
    faiss_index = None

    for i in tqdm(
        range(0, len(docs_processed), batch_size),
        desc="🔨 Building FAISS index",
    ):
        batch_docs = docs_processed[i : i + batch_size]

        if faiss_index is None:
            # First batch → create index
            faiss_index = FAISS.from_documents(
                batch_docs,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
            )
        else:
            # Next batches → add to index
            faiss_index.add_documents(batch_docs)

    # 6️⃣ Save index
    os.makedirs(index_path, exist_ok=True)
    faiss_index.save_local(index_path)

    print(f"✅ FAISS index saved to {index_path}")

    return faiss_index
