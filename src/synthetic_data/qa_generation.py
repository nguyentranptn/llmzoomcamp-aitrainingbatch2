import random
from tqdm import tqdm
from src.synthetic_data.critique_prompts import QA_GENERATION_PROMPT
from src.utils.llm_client import HFInferenceClient


def generate_synthetic_qa(
    docs_processed,
    llm_client: HFInferenceClient,
    n_generations: int = 100,
    max_answer_length: int = 300,
):
    """
    Generate synthetic (question, answer) pairs from document chunks
    """

    outputs = []
    sampled_docs = random.sample(
        docs_processed,
        min(len(docs_processed), n_generations)
    )

    print(f"Sampled documents: {len(sampled_docs)}")

    for doc in tqdm(sampled_docs, desc="Generating QA pairs"):
        try:
            prompt = QA_GENERATION_PROMPT.format(
                context=doc.page_content
            )

            raw_output = llm_client.invoke(prompt)

            question = (
                raw_output
                .split("Factoid question: ")[-1]
                .split("Answer: ")[0]
                .strip()
            )
            answer = raw_output.split("Answer: ")[-1].strip()

            if len(answer) > max_answer_length:
                continue

            outputs.append(
                {
                    "context": doc.page_content,
                    "question": question,
                    "answer": answer,
                    "source_doc": doc.metadata.get("source"),
                }
            )

        except Exception as e:
            print(f"Failed to generate QA pair: {e}")
            break

    print(f"Generated {len(outputs)} QA pairs")

    return outputs
