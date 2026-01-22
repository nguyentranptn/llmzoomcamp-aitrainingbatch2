Author: Nguyen Thanh Truc - AI team

Project overview

- This module is developed based on the provided base code and my own research results to build a complete RAG Evaluation Labwork module.
- The goal of this project is to systematically evaluate different RAG (Retrieval-Augmented Generation) configurations using:
  - Synthetic QA generation
  - Automated LLM-based evaluation (LLM-as-a-Judge)
  - Quantitative comparison and visualization of RAG performance

Installation (Quick start)

# Create virtual environment

python -m venv .venv

# Activate environment

# Windows

.venv\Scripts\activate

# Linux / macOS

source .venv/bin/activate

# Install dependencies

pip install -r requirements.txt

Project structure:

- .
- ├── main.py # Main orchestrator to run the entire pipeline
- ├── output/ # Stores evaluation results (JSON files)
- ├── data/ # Stores FAISS indexes and embedding outputs
- └── src/ # Core modules
  - ├── analysis/ # Functions for result analysis and visualization
  - ├── chunking/ # Document chunking utilities
  - ├── embeddings/ # Embedding and FAISS index building
  - ├── evaluation/ # LLM-based evaluation logic
  - ├── experiments/ # Run and compare multiple RAG configurations
  - ├── ingestion/ # Dataset loading and preprocessing
  - ├── rag/ # Core RAG pipeline implementation
  - ├── synthetic_data/ # Synthetic QA generation and filtering
  - └── utils/ # Shared utility functions

Workflow:

1. Data Preparation:
   - Load dataset using load_hf_docs()
   - Convert raw data into LangChainDocument format using build_raw_knowledge_base()
2. Phase 1 – Synthetic Data Generation (Optional): this phase can be skipped if a prepared evaluation dataset is already available.
   - Split documents into chunks using split_documents()
     - Generate synthetic QA pairs with generate_synthetic_qa()
     - Evaluate generated questions based on three criteria using critique_and_score():
       - Groundedness – whether the question is grounded in the given context
       - Relevance – whether the question is useful and meaningful
       - Standalone – whether the question can stand independently without extra context
     - Build the final evaluation dataset using build_eval_dataset()
3. Phase 2 – RAG Configuration Evaluation:
   3.1. Initialize Models

- Reader LLM (for answer generation)
- Evaluator LLM (for automatic scoring, e.g. GPT-4)

  3.2. Run RAG Experiments (run_experiments())

- For each RAG configuration:
  - Chunk and embed the knowledge base using load_or_build_faiss_index()
  - Run the RAG pipeline:
    - Retrieve documents using vector search
    - (Optional) Rerank documents with a CrossEncoder
    - Build context
    - Generate answer with the Reader LLM
  - Save outputs to a JSON file

  - Each configuration produces one JSON file containing:
    - Question
    - Ground-truth answer (true_answer)
    - Source document
    - Generated answer (generated_answer)
    - Retrieved documents (retrieved_docs)
    - Test configuration (test_setting)
    - LLM evaluation score (eval_score_GPT4)
    - LLM feedback (eval_feedback_GPT4)

  3.3. Automatic Evaluation

- Evaluate answers using an LLM-as-a-Judge approach (GPT-4)
- Due to token cost constraints, the number of evaluated samples is limited

4. Result Analysis and Visualization

- Clean and normalize evaluation scores using normalize_eval_scores()
- Compute average scores per RAG configuration using compute_average_scores()
- Visualize performance comparison using plot_rag_scores()

Questions After the Labwork

1. Will the team receive a shared base code template when starting a project?
2. If model training is too heavy for local machines, should we use platforms such as Google Colab for training and experimentation?
3. How are API keys and quotas managed during the project?
   - Are there predefined limits?
   - Should we explicitly control token usage inside the code?
4. If possible, I would greatly appreciate any feedback or suggestions to improve this module and my approach.

Acknowledgment
Thank you very much for your guidance and support.
