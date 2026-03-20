from datasets import load_dataset

def load_data():
    dataset = load_dataset("glue", "sst2")
    return dataset["validation"]
