import os
from ragatouille import RAGPretrainedModel
from sentence_transformers import CrossEncoder

from src.embeddings.build_faiss_index import load_or_build_faiss_index
from src.rag.run_rag_test import run_rag_tests
from src.evaluation.evaluate_answers import evaluate_answers
from src.utils.serialize_document import safe_filename


def run_experiments(
    RAW_KNOWLEDGE_BASE,
    eval_dataset,
    READER_LLM,
    READER_MODEL_NAME,
    eval_chat_model,
    evaluator_name,
    evaluation_prompt_template,
):
    """Runs RAG experiments with different configurations."""

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    for chunk_size in [200]:  # Extend as needed
        for embeddings in ["thenlper/gte-small"]:
            for rerank in [True, False]:

                settings_name = (
                    f"chunk:{chunk_size}_"
                    f"embeddings:{embeddings.replace('/', '~')}_"
                    f"rerank:{rerank}_"
                    f"reader-model:{READER_MODEL_NAME}"
                )
                safe_settings_name = safe_filename(settings_name)
                output_file_name = f"{output_dir}/rag_{safe_settings_name}.json"

                print("=" * 80)
                print(f"Running experiment: {settings_name}")

                print("Loading knowledge base embeddings...")
                knowledge_index = load_or_build_faiss_index(
                    RAW_KNOWLEDGE_BASE,
                    chunk_size=chunk_size,
                    embedding_model_name=embeddings,
                )

                print("Running RAG...")
                reranker = (
                    CrossEncoder(
                        "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        device="cpu",
                    )
                    if rerank
                    else None
                )

                run_rag_tests(
                    eval_dataset=eval_dataset,
                    llm=READER_LLM,
                    knowledge_index=knowledge_index,
                    output_file=output_file_name,
                    reranker=reranker,
                    verbose=False,
                    test_settings=settings_name,
                )

                print("Running evaluation...")
                evaluate_answers(
                    answer_path=output_file_name,
                    eval_chat_model=eval_chat_model,
                    evaluator_name=evaluator_name,
                    evaluation_prompt_template=evaluation_prompt_template,
                    max_eval=15
                )
