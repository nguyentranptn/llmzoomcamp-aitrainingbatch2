import pandas as pd
import datasets


def load_reference_scores(
    dataset_name: str = "m-ric/rag_scores_cookbook",
    split: str = "train",
) -> pd.Series:
    """
    Load reference RAG benchmark scores as a pandas Series.
    """
    ds = datasets.load_dataset(dataset_name, split=split)
    return pd.Series(ds["score"], index=ds["settings"])
