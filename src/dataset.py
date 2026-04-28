
from datasets import load_dataset
# Download the training and test splits from the SQuAD-it dataset
#!wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
#!wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz

# Load the local dataset
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(f"SQuAD_it dataset:\n {squad_it_dataset}")

# Load a remote dataset
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")