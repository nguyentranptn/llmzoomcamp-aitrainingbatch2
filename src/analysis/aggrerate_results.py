import glob
import json
import os
import pandas as pd
from typing import List


def load_experiment_results(output_dir: str = "./output") -> pd.DataFrame:
    """
    Load and aggregate all RAG experiment results into a single DataFrame.
    Each row corresponds to one question-answer evaluation.
    """

    outputs: List[pd.DataFrame] = []

    for file_path in glob.glob(f"{output_dir}/*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        name = os.path.basename(file_path).replace(".json", "")
        name = name.replace("rag_chunk_200_embeddings_thenlper~gte-small_", "")
        name = name.replace("_reader-model__zephyr-7b-beta.json", "")
        df["settings"] = name
        outputs.append(df)

    if not outputs:
        raise ValueError("No experiment results found.")

    return pd.concat(outputs, ignore_index=True)

def normalize_eval_scores(
    df: pd.DataFrame,
    score_column: str = "eval_score_GPT4",
) -> pd.DataFrame:
    """
    Normalize evaluation scores from 1–5 to 0–1.
    Invalid or missing scores default to 1 (worst score).
    """

    df = df.copy()

    df=df[df[score_column].notna()]

    def coerce_score(x):
        if isinstance(x, (int, float)) and 1 <= x <= 5:
            return int(x)

        if isinstance(x, str) and x.strip().isdigit():
            v = int(x.strip())
            if 1 <= v <= 5:
                return v

        # Fallback: worst score
        return 1
    
    df=df[df[score_column].notna()]

    df[score_column] = df[score_column].apply(coerce_score)

    # Normalize 1–5 -> 0–1
    df[score_column] = (df[score_column] - 1) / 4 * 100

    return df


def compute_average_scores(
    df: pd.DataFrame,
    score_column: str = "eval_score_GPT4",
) -> pd.Series:
    """
    Compute average evaluation score per RAG setting.
    """
    return (
        df.groupby("settings")[score_column]
        .mean()
        .sort_values(ascending=False)
    )
