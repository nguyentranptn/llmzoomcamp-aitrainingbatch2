def main():
    USE_REMOTE_EVAL_DATASET = True  # Allow to skip building eval dataset

    from datasets import load_dataset
    from src import config
    from src.utils.llm_client import HFInferenceClient
    from src.ingestion.documents import build_raw_knowledge_base
    from src.ingestion.load_dataset import load_hf_docs
    from src.ingestion.split_document import split_documents

    from src.synthetic_data.qa_generation import generate_synthetic_qa
    from src.synthetic_data.qa_filtering import critique_and_score, build_eval_dataset

    from src.evaluation.evaluate_answers import build_openai_evaluator
    from src.experiments.run_rag_experiments import run_experiments
    from src.evaluation.evaluation_prompt import evaluation_prompt_template

    from src.analysis.aggrerate_results import (
        load_experiment_results,
        compute_average_scores,
        normalize_eval_scores,
    )
    from src.analysis.ploy_scores import plot_rag_scores

    print("📚 Loading dataset...")
    ds = load_hf_docs()
    RAW_KNOWLEDGE_BASE = build_raw_knowledge_base(ds)

    # ===== 1. Load / Build eval dataset =====
    if USE_REMOTE_EVAL_DATASET:
        eval_dataset = load_dataset(
            "m-ric/huggingface_doc_qa_eval",
            split="train"
        )
    else:
        processed_docs = split_documents(
            RAW_KNOWLEDGE_BASE,
            chunk_size=2000,
            chunk_overlap=200,
            add_start_index=True,
        )

        llm_client = HFInferenceClient(
            repo_id=config.LLM_REPO_ID,
        )

        generated_qa = generate_synthetic_qa(processed_docs, llm_client)
        evaluations = critique_and_score(generated_qa, llm_client)
        eval_dataset = build_eval_dataset(evaluations)

    # ===== 2. Run experiments =====
    # print("🔎 Checking existing experiment results...")
    # experiment_results = load_experiment_results()

    # if experiment_results is not None and not experiment_results.empty:
    #     print("✅ Found existing results. Skipping RAG experiments...")
    # else:
    print("🧑‍🎓 Initializing Reader LLM...")
    READER_LLM = HFInferenceClient(config.READER_MODEL_REPO)

    print("🧑‍⚖️ Initializing Evaluator LLM...")
    eval_chat_model = build_openai_evaluator(
        config.EVALUATOR_MODEL_NAME,
        config.OPENAI_API_KEY,
        config.BASE_URL
    )

    print("🚀 Running RAG experiments...")
    run_experiments(
        RAW_KNOWLEDGE_BASE=RAW_KNOWLEDGE_BASE,
        eval_dataset=eval_dataset,
        READER_LLM=READER_LLM,
        READER_MODEL_NAME=config.READER_MODEL_NAME,
        eval_chat_model=eval_chat_model,
        evaluator_name=config.EVALUATOR_NAME,
        evaluation_prompt_template=evaluation_prompt_template,
    )

    # Reload results after running experiments
    experiment_results = load_experiment_results()

    # ===== 3. Analyze results =====
    print("📊 Analyzing results...")
    normalized_results = normalize_eval_scores(experiment_results)
    print(normalized_results["eval_score_GPT4"].describe())
    average_scores = compute_average_scores(normalized_results)

    print("📈 Average scores:")
    print(average_scores)
    plot_rag_scores(average_scores)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
