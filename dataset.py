from datasets import load_dataset


ds = load_dataset("archit11/deepwiki-16k")
print(ds["train"][0])