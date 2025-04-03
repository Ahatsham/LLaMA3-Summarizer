from datasets import load_dataset

def get_dataset():
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    return dataset["train"], dataset["validation"]
