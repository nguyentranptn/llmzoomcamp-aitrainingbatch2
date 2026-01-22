import os
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
READER_MODEL_REPO = "HuggingFaceH4/zephyr-7b-beta"
READER_MODEL_NAME = "zephyr-7b-beta"
READER_MODEL_KWARGS = {
    "max_new_tokens": 512,
    "top_k": 30,
    "temperature": 0.1,
    "repetition_penalty": 1.03,
}
EVALUATOR_MODEL_NAME = "gpt-4.1"
EVALUATOR_NAME = "GPT4"
LLM_REPO_ID = "Qwen/Qwen3-4B-Instruct-2507"

DATA_DIR = "data"
INDEX_DIR = "data/indexes"
RESULTS_DIR = "data/results"
DEFAULT_EMBEDDING_MODEL = "thenlper/gte-small"
DEFAULT_INDEX_PATH = "./data/indexes/"
OUTPUT_DIR = "./output"
