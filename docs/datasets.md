# :hugs: The Datasets Library
## :pushpin: How to work with local and remote datasets?

| Data format | Loading script | Example |
| :--- | :--- | :--- |
| .csv .tsv | csv | `load_dataset("csv", data_files="my_file.csv")` |
| .txt | text | `load_dataset("text", data_files="my_file.txt")` |
| .json .jsonl | json | `load_dataset("json", data_files="my_file.jsonl")` |
| pkl | pandas | `load_dataset("pandas", data_files="my_dataframe.pkl")` |

:bulb: A .tsv file can be loaded by specying the `delimiter="\t"` argument in `load_dataset()`.

For example, to load the [SQuAD_it](https://github.com/crux82/squad-it/) dataset:
```python
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

Alternativelly, we could load remotelly the SQuAD_it dataset:
```python
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```
:bulb: When loading a .json file, we need to specify the `field` where the dataset is stored.

By default, loading local files creates a `DatasetDict` object, which shows the number of rows (`num_rows`) and columns (`features`).

:bulb: With the train and test splits in a single `DatasetDict`, we can apply `Dataset.map()` functions across the splits.

## :pushpin: How to drop rows with empty values in a dataset?
We can use a ***lambda function*** with `Dataset.filter()` to drop rows with empty values.
```python
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)
```

## :pushpin: How to rename the column names of a dataset?
After loading a dataset, we can use the `DatasetDict.rename_colum()` to rename the columns.

For example, in the [Drug Review Dataset](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29), we can rename the `Unamed: 0` column to `patient_id`:
```python
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
```

In addition, we can normalize the column names to lowercase or uppercase using `Dataset.map()` by writing a function:
```python
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

drug_dataset.map(lowercase_condition)
```

## :pushpin: How to create a column?
We can add a column to the dataset that contains the number of words in each the column `review`.

To count the number of words, we can write `compute_review_length()` to split each text by whitespace and calculate the length. Then, we can apply
`compute_review_length()` to all the rows using `Dataset.map()`.
```python
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}
drug_dataset = drug_dataset.map(compute_review_length)
```

:bulb: Alternatively, we could use `Dataset.add_colum()`.

## :pushpin: How to sort a column?
We can sort a column using `Dataset.sort()`:
```python
drug_dataset["train"].sort("review_length")[:3]
```

## How to drop rows following a rule?
We can use `Dataset.filter()` to remove the reviews contianing less than 30 words, because they couldn't be informative for predicting the condition. 
```python
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)
```

Also, we can use `Dataset.map()` to unescape all the HTML characters in our corpus:
```python
drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})
```

:bulb: The `Dataset.map()` argument `batched=True` results in a batch of examples to the map function, speeding up by processing several elements at the same time. This will be important to unlock the speed of tokenizers.

## :pushpin: How to split the dataset?
The `Dataset.train_test_split()` can be used to split the training set into `train` and `validation` splits:
```python
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
```

## :pushpin: How to save a dataset?
| Data format | Function                 |
| :-----------| :------------------------|
| Arrow       | `Dataset.save_to_disk()` |
| .csv        | `Dataset.to_csv()`       |
| .json       | `Dataset.to_json()`      |

For .csv and .json formats, we need to store each split as a separate file. We can iterate over the keys and values in `DatasetDict`:
```python
for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")
```

Then, we can load the .json files as:
```python
data_files = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews-test.jsonl",
}
drug_dataset_reloaded = load_dataset("json", data_files=data_files)
```

## :pushpin: Creating your own dataset
### Getting the dataset
A conventient way to get a datase is via the `requests` library, which make HTTP requests in Python.

In the following example, we use `requests.get()` to create a corpus from the `GitHub Issues`.

```python
import time
import math
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm


def fetch_issues(
    owner="huggingface",
    repo="datasets",
    num_issues=10_000,
    rate_limit=5_000,
    issues_path=Path("."),
):
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 100  # Number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        # Query with state=all to get both open and closed issues
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []  # Flush batch for next time period
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(
        f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl"
    )
```

Afterwards. we can load the dataset locally using `Dataset.load_dataset()`.
```python
issues_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")
```

### Uploading the dataset to the Hugging Face Hub
We can use `push_to_hub()` to push a dataset with an authentication token.
```python
from huggingface_hub import notebook_login
notebook_login()
```
After logging into the Hugging Face Hub, we can upload the dataset by running:
```python
from huggingface_hub import notebook_login

notebook_login()
```

Then, anyone can download the dataset by providing `load_dataset()` with the repository as the `path` argument:
```python
remote_dataset = load_dataset("lewtun/github-issues", split="train")
remote_dataset
```

```python
Dataset({
    features: ['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'pull_request', 'body', 'performed_via_github_app', 'is_pull_request'],
    num_rows: 2855
})
```

In addition, we can create a ***dataset card*** to explain how the corpus was created.