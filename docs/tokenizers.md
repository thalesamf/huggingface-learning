# :hugs: The Tokenizers Library
## :pushpin: How to train a new tokenizer?
If the langague model is not available in the language of interst, or if the corpus is very different form the one the language model was trained on, we can retrain the model from scratch using a tokenizer adapted to the dataset of interest, which requires a new tokenizer on your dataset.

The three most common subword tokenization algorithms used with Transformers (i.e., ***Byte-Pair Encoding [BPE], WordPiece, and Unigram***)

Training a tokenizer is deterministic and involves the identification of which subwords are the best to pick for a given corpus, and the exact rules to pick them depends on the tokenization algorithm.

### Assembling a corpus
The `AutoTokenizer.train_from_iterator()` can be used to train a new tokenizer with the same characteristics as an existing one.

In the example below, we are going to train GPT-2 from scracth, but using Python source code instead of English, from [CodeSearchNet](https://huggingface.co/datasets/code_search_net), which contains millions of functions from GitHub.

We'll use `load_dataset()` to download and cache the dataset:
```python
from datasets import load_dataset
raw_datasets = load_dataset("code_search_net", "python")
```

In the training split, we can see that the dataset separates ***docstrings*** from ***code*** and suggests a tokenization for both. In our example, we'll use the `whole_func_string` column to train the tokenizer.

```python
raw_datasets["train"]
```
```python
Dataset({
    features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 
      'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 
      'func_code_url'
    ],
    num_rows: 412178
})
```

Initially, we need to ***transform the dataset into an iterator of lists of texts***, enabling the tokenizer to speed up, instead of processing individual texts one by one, and avoid having everything in memory at once. 

The following lines creates an object that we can use in a Python `for`loop. However, it can obly be used once.

```python
training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)
```

Therefore, we define a funciton that returns a generator instead:
```python
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )


training_corpus = get_training_corpus()
```

### Training
After preparing the corpus in the form of an interator of batches of texts, we can train the new tokenizer. Instead of starting entirely from scract, we'll load the original tokenizer from the model (i.e., GPT-2), thus we won't have to specify anything about the tokenization algorithm.
```python
from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

To train the new tokenizer, we'll use `train_new_from_interator()`:
```python
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
```

After training the tokenizer, we can check the tokenization:
```python
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = tokenizer.tokenize(example)
tokens
```

```python
['def', 'Ġadd', '_', 'numbers', '(', 'a', ',', 'Ġb', '):', 'ĊĠĠĠ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`',
 'a', '`', 'Ġand', 'Ġ`', 'b', '`."""', 'ĊĠĠĠ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
```

We can see the special tokens `Ġ` and `Ċ` that denotes spaces and newlines, respectively, and the learned tokens `ĊĠĠĠ` and that represents an identation `Ġ"""` that represents an indentation and the start of a docstring, respectively. In addition, the tokenizer correctly split the function name on `_`.


### Saving the tokenizer
To save the new tokenizer, we'll use `save_pretrained()`, which will create a new folder named ***code-search-net-tokenizer*** containing the files the tokenizer needs to be reloaded.
```python
tokenizer.save_pretrained("code-search-net-tokenizer")
```

To share the tokenizer in the Hugging Face Hub, we login the credentials and run `push_to_hub()`. This will create a new repository in your namespace with the name `code-search-net-tokenizer`, containing the tokenizer file.

```python
from huggingface_hub import notebook_login

notebook_login()

tokenizer.push_to_hub("code-search-net-tokenizer")
```

Then, we can load the tokenizer using `from_pretrained()`:
```python
# Replace "huggingface-course" below with the actual namespace to use the tokenizer
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

```

## :pushpin: Normalization and pre-tokenization
Before splitting the text into subtokens, the tokenizer performs ***normalization*** and ***pre-tokenization***.

### Normalization
The ***normalization*** involves cleaning up, such as removing whitespaces, lowercasing, or [Unicode normalization](http://www.unicode.org/reports/tr15/).

The `tokenizer` object has an attribute called `backend_tokenizer`, which provides access to the underlying tokenizer from Tokenizers library:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))
```

```python
<class 'tokenizers.Tokenizer'>
```

Also, the `tokenizer` object has an attribute called `normalizer`, which has a `normalize_str()` method that we can use to see how the normalization is performed.
```python
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
```

```python
'hello how are u?'
```

### Pre-tokenization
Because a tokenizer cannot be trained on raw text alone, it first need to split the texts into words. For example, the ***word-based tokenizer*** splits a raw text into words on whitespace and punctuation. Then, the words will be the boundaries of the subtokens the tokenizer can learn during its training.

The `tokenizer` object has a `pre_tokenizer` attribute, which  has a method called `pre_tokenizer_str()` that allows us to see how the tokenizer peforms pre-tokenization:
```python
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

```python
[('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]
```

Since we are using a BERT tokenizer, the pre-tokenization involves splitting on whitespace and punctuation. If we use the GPT-2 tokenizer, it will split on whitespace and punctuation, but it will keep the spaces and replace them with a `Ġ` symbol, enabling it to recover the original spaces if we decode the tokens, and does not ignore the double space.
```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

```python
[('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)),
 ('?', (19, 20))]
```

Lastly, the T5 tokenizer, which is based on the SentencePiece algorithm, keeps spaces and replaces them with a specific token (`_`), but only splites on whitespace, and ignores double spaces.
```python
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

```python
[('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))]
```

## :pushpin: Tokenization Algorithms


| Model | BPE | WordPiece | Unigram |
| :--- | :--- | :--- | :--- |
| **Training** | Starts from a small vocabulary and learns rules to merge tokens | Starts from a small vocabulary and learns rules to merge tokens | Starts from a large vocabulary and learns rules to remove tokens |
| **Training step** | Merges the tokens corresponding to the most common pair | Merges the tokens corresponding to the pair with the best score based on the frequency of the pair, privileging pairs where each individual token is less frequent | Removes all the tokens in the vocabulary that will minimize the loss computed on the whole corpus |
| **Learns** | Merge rules and a vocabulary | Just a vocabulary | A vocabulary with a score for each token |
| **Encoding** | Splits a word into characters and applies the merges learned during training | Finds the longest subword starting from the beginning that is in the vocabulary, then does the same for the rest of the word | Finds the most likely split into tokens, using the scores learned during training |

[SentencePiece](https://github.com/google/sentencepiece) onsiders the text as a sequence of Unicode characters, and replaces spaces with a special character, `_`. Also, it performs ***reversible tokenization***, which allows to decode the tokens by concatenating them and replacing the `_` with spaces.

