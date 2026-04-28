
from datasets import load_dataset
# Download
#!wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
#!wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz
#!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
#!unzip drugsCom_raw.zip

# Load the local dataset
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(f"\nSQuAD_it dataset:\n {squad_it_dataset}")

# Load a remote dataset
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

# Splitting
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000)) 
print(f"\nDrug Review Dataset examples:\n {drug_sample[:3]}")
