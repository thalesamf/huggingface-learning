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

### Byte-Pair Encoding (BPE)
The BPE algorith as initially developed to compress texts, and then used for tokenization of Transformer models, including GPT, GPT-2, RoBERTa, BART, and DeBERTa.

#### Algorithm
After normalization and pre-tokenization, BPE starts by computing the unique set of words used in the corpus. Then, BPE builds the vocabulary by taking all the symbols used to write the words.

For example, if the corpus has the five words `"hug", "pug", "pun", "bun", "hugs"`, the vocabulary would contain `["b", "g", "h", "n", "p", "s", "u"]`. In real, the vocabulary will contain all the **ASCII charaters**, and some **Unicode** charaters. If the text uses a haracter that is not in the training corpus, that character will be converted to the unknown token.

:bulb: The GPT-2 and RoBERTa tokenizers don’t look at words as being written with Unicode characters, but with bytes. This way the base vocabulary has a small size (256), but every character will still be included and not end up being converted to the unknown token, which is known as **byte-level BPE**.

Ater getting the vocabulary, we add new tokens until the desired vocabulary is reached by learning ***merges**, which are rules to merge to elements of the existing vocabulary into a new one. Thus, during training the merges creates tokens with two characters and then longer subwords.

During tokenizer training, the BPE algorithm search for the most frequent pair of existing tokens (i.e., two consecutive tokens in a word), and merges the mos frequent pair.

In the previous example, consider that the words has the following frequencies in the corpus:
```python
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

During tokenizer training, each work is splitted into characters, producing a list of tokens:
```python
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

Looking at pairs, `("h", "u")` is present in the words `"hug"` and `"hugs"`, thus 15 times in the corpus. In parallel, `("u", "g")` is present 20 times in the vocabulary in the words `"hug"`. `"pug"`, and `"hugs"`.

Consequently, the first merge learned by the tokenizer is the merging `"u"` with `"g"`, producing `"ug"`, which will be added to the vocabulary:
```python
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

Now, there are pais that result in a token larger than two characters, such as the pair the pair `("h", "ug")`, which is present 15 times in the corpus. However, the most frequent pair now is `("u", "n")`, which is present 16 times in the corpus. This, the second merge learned by the tokenizer is the merging `"u"` with `"m"`, producing `"un"`, which will be loaded to the vocabulary:
```python
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug", "un"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
```

Continuing, we reach the desired vocabulary size. Because the word `"mug"` contain the character `"m"` that is not present in the vocabulary, it will be tokenized as `["[UNK]", "ug"]`.

#### Implementation
In the example below, we created a corpus with few sequences:
```python
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
```

Next, we load the BPE tokenizer from the GPT-2 model to pre-tokenize the corpus into words:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

Then, we compute the frequencies of each word in the corpus as we do the pretokenization:
```python
from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)
```

```python
defaultdict(int, {'This': 3, 'Ġis': 2, 'Ġthe': 1, 'ĠHugging': 1, 'ĠFace': 1, 'ĠCourse': 1, '.': 4, 'Ġchapter': 1,
    'Ġabout': 1, 'Ġtokenization': 1, 'Ġsection': 1, 'Ġshows': 1, 'Ġseveral': 1, 'Ġtokenizer': 1, 'Ġalgorithms': 1,
    'Hopefully': 1, ',': 1, 'Ġyou': 1, 'Ġwill': 1, 'Ġbe': 1, 'Ġable': 1, 'Ġto': 1, 'Ġunderstand': 1, 'Ġhow': 1,
    'Ġthey': 1, 'Ġare': 1, 'Ġtrained': 1, 'Ġand': 1, 'Ġgenerate': 1, 'Ġtokens': 1})
```

Then, we compute the base vocabulary formed by all the characters used in the corpus:
```python
alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

print(alphabet)
```

```python
[ ',', '.', 'C', 'F', 'H', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's',
  't', 'u', 'v', 'w', 'y', 'z', 'Ġ']
```

We also add the special tokens used by the model at the beginning of that vocabulary. In the case of GPT-2, the only special token is `"<|endoftext|>"`:
```python
vocab = ["<|endoftext|>"] + alphabet.copy()
```

Then, we need to split each word into individual characters:
```python
splits = {word: [c for c in word] for word in word_freqs.keys()}
```

Next, we compute the frequency of each pair:
```python
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

pair_freqs = compute_pair_freqs(splits)
```
Then, we need to merge the pair of tokens from the `splits` dictionary:
```python
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
```

Now, we loop until the tokenizer have learned all the merges we want aiming the vocabulary size:
```python
vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])
```

Consequently, the tokenizer has learned 19 merge rules:
```python
print(merges)
```

```python
{('Ġ', 't'): 'Ġt', ('i', 's'): 'is', ('e', 'r'): 'er', ('Ġ', 'a'): 'Ġa', ('Ġt', 'o'): 'Ġto', ('e', 'n'): 'en',
 ('T', 'h'): 'Th', ('Th', 'is'): 'This', ('o', 'u'): 'ou', ('s', 'e'): 'se', ('Ġto', 'k'): 'Ġtok',
 ('Ġtok', 'en'): 'Ġtoken', ('n', 'd'): 'nd', ('Ġ', 'is'): 'Ġis', ('Ġt', 'h'): 'Ġth', ('Ġth', 'e'): 'Ġthe',
 ('i', 'n'): 'in', ('Ġa', 'b'): 'Ġab', ('Ġtoken', 'i'): 'Ġtokeni'}
```

In addition, the vocabulary is compsed by the special token, the initial alphabet, and all the merges:
```python
print(vocab)
```

```python
['<|endoftext|>', ',', '.', 'C', 'F', 'H', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o',
 'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', 'Ġ', 'Ġt', 'is', 'er', 'Ġa', 'Ġto', 'en', 'Th', 'This', 'ou', 'se',
 'Ġtok', 'Ġtoken', 'nd', 'Ġis', 'Ġth', 'Ġthe', 'in', 'Ġab', 'Ġtokeni']
```

Then, we can tokenize a new text by pre-tokenizing, splitting, and applying the merge rules:
```python
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])

print(tokenize("This is not a token."))
```

```python
['This', 'Ġis', 'Ġ', 'n', 'o', 't', 'Ġa', 'Ġtoken', '.']
```

:bulb: Because GPT-2 doesn't have an unknown token, it’s impossible to get an unknown character when using byte-level BPE, but this could happen here because we did not include all the possible bytes in the initial vocabulary.

### WordPiece
The WordPiece is the tokenization algorithm developed to pretrain **BERT**, and has been reused in few Transfromers models based on BERT, such as DistilBERT, Funnel Transformers, and MPNET.

#### Algorithm
As BPE, WordPiece starts from a small vocabulary including the special tokens used by the model, and the intial alphabet. As WrodPiece identifies subwords by addind a prefix (`##`), each word is initially splitted y adding that prefix to all the charcters inside the word. For example, `"word"` is splitted to `w ##o ##r ##d`.

Consequently, the initial alphabet contains all characters present in the beggining of a word and the characters present inside a word preced by the WordPiece prefix (e.g., `##`).

As BPE, WordPiece learns merge rules, but instead of selecting the most frequent pair, WordPiece computes a score for each pair.

$score=(freq_of_pair)/(freq_of_first_element×freq_of_second_element)$

 By dividing the frequency of each pair (`freq_of_pair`) by the product between the frequencies of each of the pairs (`freq_of_first_element×freq_of_second_element`), the algorithm prioritize the merging of pairs where the individual parts are less frequent in the vocabulary.

 From the vocabulary that we used in the BPE example, the splits would be:
 ```python
 ("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

```python
 ("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##g" "##s", 5)
 ```

 Thus, the initial vocabulary would be `["b", "h", "p", "##g", "##n", "##s", "##u"]`.

 One difference between WordPiece and BPE is that it only saves the final vocabulary, not the merged rules learned. For instance, if we use the vocabulary learned in the example above, for the word `"hugs"` the longest subword starting from the beginning that is inside the vocabulary is `"hug"`, so we split there and get `["hug", "##s"]`. We then continue with `"##s"`, which is in the vocabulary, so the tokenization of `"hugs"` is `["hug", "##s"]`. In contrast, with BPE `"hugs"` would be tokenized as `["hu", "##gs"]`.

 When the tokenization gets to a stage where it’s not possible to find a subword in the vocabulary, the whole word is tokenized as unknown `["[UNK]"]`.

 #### Implementation

Using the same corpus as in BPE, we first need to pre-tokenize the corpus into words.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

Then, we compute the frequencies of each word in the corpus:
```python
from collections import defaultdict

word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

word_freqs
```

```python
defaultdict(
    int, {'This': 3, 'is': 2, 'the': 1, 'Hugging': 1, 'Face': 1, 'Course': 1, '.': 4, 'chapter': 1, 'about': 1,
    'tokenization': 1, 'section': 1, 'shows': 1, 'several': 1, 'tokenizer': 1, 'algorithms': 1, 'Hopefully': 1,
    ',': 1, 'you': 1, 'will': 1, 'be': 1, 'able': 1, 'to': 1, 'understand': 1, 'how': 1, 'they': 1, 'are': 1,
    'trained': 1, 'and': 1, 'generate': 1, 'tokens': 1})
```

The alphabet is the unique set composed of all the first letters of words, and all the other letters that appear in words prefixed by `##`:
```python
alphabet = []
for word in word_freqs.keys():
    if word[0] not in alphabet:
        alphabet.append(word[0])
    for letter in word[1:]:
        if f"##{letter}" not in alphabet:
            alphabet.append(f"##{letter}")

alphabet.sort()
alphabet

print(alphabet)
```

```python
['##a', '##b', '##c', '##d', '##e', '##f', '##g', '##h', '##i', '##k', '##l', '##m', '##n', '##o', '##p', '##r', '##s',
 '##t', '##u', '##v', '##w', '##y', '##z', ',', '.', 'C', 'F', 'H', 'T', 'a', 'b', 'c', 'g', 'h', 'i', 's', 't', 'u',
 'w', 'y']
```

We also add the special tokens used by the model at the beginning of that vocabulary. In the case of BERT, it’s the list `["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]`:
```python
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
```

Next, we split each word with all the letters that are not the first prefixed with `##`:
```python
splits = {
    word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
    for word in word_freqs.keys()
}
```

Then, we compute the score of each pair:
```python
def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores
pair_scores = compute_pair_scores(splits)
```
We need to apply that merge in our splits dictionary:
```python
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
```

Lastly, we loop until we have learned all the merges we want:
```python
vocab_size = 70
while len(vocab) < vocab_size:
    scores = compute_pair_scores(splits)
    best_pair, max_score = "", None
    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    splits = merge_pair(*best_pair, splits)
    new_token = (
        best_pair[0] + best_pair[1][2:]
        if best_pair[1].startswith("##")
        else best_pair[0] + best_pair[1]
    )
    vocab.append(new_token)

print(vocab)
```

```python
['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '##a', '##b', '##c', '##d', '##e', '##f', '##g', '##h', '##i', '##k',
 '##l', '##m', '##n', '##o', '##p', '##r', '##s', '##t', '##u', '##v', '##w', '##y', '##z', ',', '.', 'C', 'F', 'H',
 'T', 'a', 'b', 'c', 'g', 'h', 'i', 's', 't', 'u', 'w', 'y', 'ab', '##fu', 'Fa', 'Fac', '##ct', '##ful', '##full', '##fully',
 'Th', 'ch', '##hm', 'cha', 'chap', 'chapt', '##thm', 'Hu', 'Hug', 'Hugg', 'sh', 'th', 'is', '##thms', '##za', '##zat',
 '##ut']
```

To tokenize a new text, we pre-tokenize it, split it, then apply the tokenization algorithm on each word. That is, we look for the biggest subword starting at the beginning of the first word and split it, then we repeat the process on the second part, and so on for the rest of that word and the following words in the text:
```python
def encode_word(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens
```

As result, we can write a function that tokenizes a text:
```python
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    encoded_words = [encode_word(word) for word in pre_tokenized_text]
    return sum(encoded_words, [])

print(tokenize("This is the Hugging Face course!"))
```

```python
['Th', '##i', '##s', 'is', 'th', '##e', 'Hugg', '##i', '##n', '##g', 'Fac', '##e', 'c', '##o', '##u', '##r', '##s',
 '##e', '[UNK]']
```

#### SentencePiece
[SentencePiece](https://github.com/google/sentencepiece) considers the text as a sequence of Unicode characters, and replaces spaces with a special character, `_`. Also, it performs ***reversible tokenization***, which allows to decode the tokens by concatenating them and replacing the `_` with spaces.

