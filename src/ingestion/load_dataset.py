import datasets

def load_hf_docs():
    return datasets.load_dataset("m-ric/huggingface_doc", split="train")