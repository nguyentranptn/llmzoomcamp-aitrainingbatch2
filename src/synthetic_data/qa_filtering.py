from tqdm import tqdm
from datasets import Dataset as datasets
import pandas as pd
from src.utils.llm_client import HFInferenceClient
from src.synthetic_data.critique_prompts import (
    QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT,
    QUESTION_RELEVANCE_CRITIQUE_PROMPT,
    QUESTION_STANDALONE_CRITIQUE_PROMPT,
)

def critique_and_score(outputs, llm_client: HFInferenceClient):
    print("Critiquing and scoring outputs...")
    for output in tqdm(outputs):
        try:
            evaluations = {
                "groundedness": llm_client.invoke(
                    QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT.format(
                        context=output["context"],
                        question=output["question"],
                    ),
                ),
                "relevance": llm_client.invoke(
                    QUESTION_RELEVANCE_CRITIQUE_PROMPT.format(
                        question=output["question"]
                    ),
                ),
                "standalone": llm_client.invoke(
                    QUESTION_STANDALONE_CRITIQUE_PROMPT.format(
                        question=output["question"]
                    ),
                ),
            }

            for criterion, evaluation in evaluations.items():
                score = int(evaluation.split("Total rating: ")[-1].strip())
                rationale = evaluation.split("Evaluation: ")[1].split(
                    "Total rating:"
                )[0]

                output[f"{criterion}_score"] = score
                output[f"{criterion}_eval"] = rationale

        except Exception as e:
            print(e)
            break

    return outputs

def build_eval_dataset(outputs, min_score=4):
    df = pd.DataFrame(outputs)
    filtered = df[
        (df["groundedness_score"] >= min_score)
        & (df["relevance_score"] >= min_score)
        & (df["standalone_score"] >= min_score)
    ]
    return datasets.from_pandas(filtered, preserve_index=False)
