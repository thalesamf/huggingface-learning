
from datasets import load_dataset
#!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
#!unzip drugsCom_raw.zip

# Load dataset
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

# Data postprocessing
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None) # Drop empty rows

drug_dataset = drug_dataset.rename_column(original_column_name="Unnamed: 0", new_column_name="patient_id") # Rename column

drug_dataset.map(lowercase_condition) # Normalize

drug_dataset = drug_dataset.map(compute_review_length) # Compute the length of the reviews

drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30) # Filter for reviewers with more than 30 words

drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])}) # Drop rows with HTML characters

# Splitting
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
drug_dataset_clean["test"] = drug_dataset["test"]

for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")