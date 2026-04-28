## :hugs: The Datasets Library
### :pushpin: How to work with local and remote datasets?

| Data format | Loading script | Example |
| :--- | :--- | :--- |
| .csv .tsv | csv | `load_dataset("csv", data_files="my_file.csv")` |
| .txt | text | `load_dataset("text", data_files="my_file.txt")` |
| .json .jsonl | json | `load_dataset("json", data_files="my_file.jsonl")` |
| pkl | pandas | `load_dataset("pandas", data_files="my_dataframe.pkl")` |

For example, to load locally the downloaded SQuAD_it dataset:
```
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

Alternativelly, we could load remotelly the SQuAD_it dataset:
```
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```
:bulb: When loading a ```.json``` file, we need to specify the ``field ``` where the dataset is stored.

By default, loading local files creates a ```DatasetDict``` object, which shows the number of rows (```num_rows```) and columns (```features```).

:bulb: With the train and test splits in a single ```DatasetDict```, we can apply ```Dataset.map()``` functions across the splits.

### :pushpin: How to slice a dataset?

