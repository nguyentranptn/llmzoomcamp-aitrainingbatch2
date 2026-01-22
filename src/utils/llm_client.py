from src import config
import json
from huggingface_hub import InferenceClient
from langchain_community.llms import HuggingFaceHub


# ========= Phase 1: HF Inference Client =========
class HFInferenceClient:
    def __init__(self, repo_id: str, timeout: int = 120):
        self.client = InferenceClient(
            model=repo_id,
            timeout=timeout,
            token=config.HF_API_TOKEN
        )

    def invoke(self, prompt: str, max_new_tokens: int = 1000) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_new_tokens

        )
        return response.choices[0].message["content"]

