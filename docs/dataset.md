## :hugs: The Datasets Library
### :pushpin: How to work with local and remote datasets?

| Data format | Loading script | Example |
| :--- | :--- | :--- |
| .csv .tsv | csv | `load_dataset("csv", data_files="my_file.csv")` |
| .txt | text | `load_dataset("text", data_files="my_file.txt")` |
| .json .jsonl | json | `load_dataset("json", data_files="my_file.jsonl")` |
| pkl | pandas | `load_dataset("pandas", data_files="my_dataframe.pkl")` |

:bulb: A .tsv file can be loaded by specying the `delimiter` argument in `load_dataset()`.

For example, to load locally the downloaded SQuAD_it dataset:
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

### :pushpin: How to drop rows with empty values in a dataset?
We can use a ***lambda function*** with `Dataset.filter()` to drop rows with empty values.
```python
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)
```

### :pushpin: How to rename the column names of a dataset?
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

### :pushpin: How to create a column?
We can add a column to the dataset that contains the number of words in each the column `review`.

To count the number of words, we can write `compute_review_length()` to split each text by whitespace and calculate the length. Then, we can apply
`compute_review_length()` to all the rows using `Dataset.map()`.
```python
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}
drug_dataset = drug_dataset.map(compute_review_length)
```

:bulb: Alternatively, we could use `Dataset.add_colum()`.

### :pushpin: How to sort a column?
We can sort a column using `Dataset.sort()`:
```python
drug_dataset["train"].sort("review_length")[:3]
```

### How to drop rows following a rule?
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

### :pushpin: How to split the dataset?
The `Dataset.train_test_split()` can be used to split the training set into `train` and `validation` splits:
```python
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
```

### :pushpin: How to save a Dataset?
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

### :pushpin: How to work with big datasets?
