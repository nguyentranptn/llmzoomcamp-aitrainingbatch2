import json
import os
from typing import Optional

from tqdm import tqdm
import datasets

from ragatouille import RAGPretrainedModel
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.llms import LLM

from src.rag.rag_pipeline import answer_with_rag
from src.utils.sanitize import sanitize_filename
from src.utils.serialize_document import serialize_docs


def run_rag_tests(
    eval_dataset: datasets.Dataset,
    llm: LLM,
    knowledge_index: VectorStore,
    output_file: str,
    reranker: Optional[RAGPretrainedModel] = None,
    verbose: bool = True,
    test_settings: Optional[str] = None,
):
    """
    Run RAG on an evaluation dataset and save generated answers.

    This function:
    - Iterates over eval questions
    - Generates answers using RAG
    - Saves results incrementally (checkpointing)
    """

        # --- Make output file Windows-safe ---
    output_file = sanitize_filename(output_file)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    # Load existing results if present
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            outputs = json.load(f)
    except Exception:
        outputs = []

    answered_questions = {item["question"] for item in outputs}

    for example in tqdm(eval_dataset, desc="Running RAG evaluation"):
        question = example["question"]

        # Skip already processed questions
        if question in answered_questions:
            continue

        answer, retrieved_docs = answer_with_rag(
            question=question,
            llm=llm,
            knowledge_index=knowledge_index,
            reranker=reranker,
        )

        if verbose:
            print("=" * 60)
            print(f"Question      : {question}")
            print(f"Generated     : {answer}")
            print(f"Ground Truth  : {example['answer']}")

        result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example.get("source_doc"),
            "generated_answer": answer,
            "retrieved_docs": serialize_docs(retrieved_docs),
        }

        if test_settings:
            result["test_settings"] = test_settings

        outputs.append(result)

        # Save after each example (safe checkpoint)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
