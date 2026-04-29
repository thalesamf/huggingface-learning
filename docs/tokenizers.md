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

### WordPiece
The WordPiece is the tokenization algorithm developed to pretrain **BERT**, and has been reused in few Transfromers models based on BERT, such as DistilBERT, Funnel Transformers, and MPNET.

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

### Unigram
[SentencePiece](https://github.com/google/sentencepiece) considers the text as a sequence of Unicode characters, and replaces spaces with a special character, `_`. Also, it performs ***reversible tokenization***, which allows to decode the tokens by concatenating them and replacing the `_` with spaces.

SentencePiece addresses the fact that not all languages use spaces to separate words. Instead, SentencePiece treats the input as a raw input stream which includes the space in the set of characters to use. Then it can use the Unigram algorithm to construct the appropriate vocabulary.

 Unigram works in the other direction: it starts from a big vocabulary and removes tokens from it until it reaches the desired vocabulary size.

 At each step of the training, the Unigram algorithm computes a loss over the corpus given the current vocabulary. Then, for each symbol in the vocabulary, the algorithm computes how much the overall loss would increase if the symbol was removed, and looks for the symbols that would increase it the least.

 From the corpus that we used in the BPE and WordPiece examples, the splits and the intiali vocabulary would be:
 ```python
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

["h", "u", "g", "hu", "ug", "p", "pu", "n", "un", "b", "bu", "s", "hug", "gs", "ugs"]
 ```

the Unigram model considers each token to be independent of the tokens before it.The probability of a given token is its frequency (the number of times we find it) in the original corpus, divided by the sum of all frequencies of all tokens in the vocabulary (to make sure the probabilities sum up to 1).

Beloware the frequencies of all possible subwords in the vocabulary:
```python
("h", 15) ("u", 36) ("g", 20) ("hu", 15) ("ug", 20) ("p", 17) ("pu", 17) ("n", 16)
("un", 16) ("b", 4) ("bu", 4) ("s", 5) ("hug", 15) ("gs", 5) ("ugs", 5)
```

To tokenize a given word, we look at all the possible segmentations into tokens and compute the probability of each according to the Unigram model. Since all tokens are considered independent, this probability is just the product of the probability of each token

$P(["p", "u", "g"]) = P("p") x P("u") x P("g")$

Therefore, the tokenization of a word with the Unigram model is the tokenization with the highest probability.

In the case of `"pug"`, below are the probabilities for each possible segmentation:
```python
["p", "u", "g"] : 0.000389
["p", "ug"] : 0.0022676
["pu", "g"] : 0.0022676
```

Thus, `"pug"` would be tokenized a `["p", "ug"]` or `["pu", "g"]`,  depending on which of those segmentations is encountered first.

At any given stage, this loss is computed by tokenizing every word in the corpus, using the current vocabulary and the Unigram model determined by the frequencies of each token in the corpus. Each word in the corpus has a score, and the loss is the negative log likelihood of those scores (i.e., the sum for all the words in the corpus of all the `-log(P(word))`).

For the corpus, the tokenization of each word with the respective score is:
```python
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

"hug": ["hug"] (score 0.071428)
"pug": ["pu", "g"] (score 0.007710)
"pun": ["pu", "n"] (score 0.006168)
"bun": ["bu", "n"] (score 0.001451)
"hugs": ["hug", "s"] (score 0.001701)
```

Thus, the loss is:
```python
10 * (-log(0.071428)) + 5 * (-log(0.007710)) + 12 * (-log(0.006168)) + 4 * (-log(0.001451)) + 5 * (-log(0.001701)) = 169.8
```

Next, we compute how removing each token affects the loss, and remove from the vocabulary the tokens.

## :pushpin: Building a tokenizer

The `Tokenizer` class has the [submodules](https://huggingface.co/docs/tokenizers/components) necessary for building a tokenizer:
* `normalizers` contains all the possible types of [`Normalizer`](https://huggingface.co/docs/tokenizers/api/normalizers)
* `pre_tokenizers` has all the possible types of [`PreTokenizer`](https://huggingface.co/docs/tokenizers/api/pre-tokenizers)
* `models` has all the possible types of [`Model`](https://huggingface.co/docs/tokenizers/api/models), such as `BPE` and `WordPiece`
* `trainers` has all the possible types of [`Trainer`](https://huggingface.co/docs/tokenizers/api/trainers) to train the model on a corpus
* `post_processors` has all the possible types of [`PostProcessor`](https://huggingface.co/docs/tokenizers/api/post-processors)
* `decoders` has all the possible types of [`Decoder`](https://huggingface.co/docs/tokenizers/components#decoders)

### Getting a corpus
We are going to use the [WikiText-2](https://huggingface.co/datasets/wikitext) dataset:
```python
from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
```

The function `get_training_corpus()` is a generator that will yield batches of 1,000 texts, which we will use to train the tokenizer.

Alternatively, we can yse the file directly:
```python
with open("wikitext-2.txt", "w", encoding="utf-8") as f:
    for i in range(len(dataset)):
        f.write(dataset[i]["text"] + "\n")
```

### Building a WordPiece tokenizer from scratch
#### 1. Instantiate a `Tokenizer` object

Let's create a `Tokenizer` with a WordPiece model:
```python
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
```

:bulb: We have to specify the `unk_token` so the model knows what to return when it encounters characters it hasn't seen before.

#### 2. Normalization
Since BERT is widely used, there is a `BertNormalizer` with the options `lowercase`, `strip_accents`, `clean_text`, which removes all control characters and replace repeating spaces with a single one. To replicate the model `bert-base-uncased` tokenizer, we can just set this normalizer:
```python
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
```

Alternatively, we can create a normalizer from scratch:
```python
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
```

#### 3. Pre-tokenization
There is a prebuilt `BertPreTokenizer`:
```python
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
```

Alternatively, we can build a pre-tokenizer from scratch that splites on whitespace:
```python
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
```

As the normalizers, we can use a `Sequence` to compose several pre-tokenizers:
```python
pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)
```

#### 3. Train the Tokenizer
When instatiating a trainer, we need to pass all the special tokens we are intended to use, otherwise it won't add them to the vocabulary:
```python
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
```

Then, we call `WordPieceTrainer` to train our tokenizer.
```python
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
```

We can train our tokenizer directly using the text file with:
```python
tokenizer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
```

Then, we can test the tokenizer on a text by calling the `encode()` method. The `encoding` obtained is an `Encoding`, which contains all the necessary outputs of the tokenizer in its various attributes: ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, and overflowing
```python
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
```

```python
['let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.']
```

#### 4. Post-processing

We need to add the `[CLS]` token at the beginning and the `[SEP]` token at the end (or after each sentence, if we have a pair of sentences), which can be done using `TemplateProcessor`, but we first need to know the IDs of the `[CLS]` and `[SEP]` tokens in the vocabulary.
```python
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)
```

```python
(2,3)
```

To write the template for the TemplateProcessor, we have to specify how to treat a single sentence and a pair of sentences. For both, we write the special tokens we want to use; the first (or single) sentence is represented by $A, while the second sentence (if encoding a pair) is represented by $B. For each of these (special tokens and sentences), we also specify the corresponding token type ID after a colon.
```python
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)
```

The last step is to add a decoder:
```python
tokenizer.decoder = decoders.WordPiece(prefix="##")
```

Then, we can save our tokenizer in a single JSON file:
```python
tokenizer.save("tokenizer.json")
```

We can then reload that file in a `Tokenizer` object with the `from_file()` method:
```python
tokenizer = Tokenizer.from_file("tokenizer.json")
```

To use the tokenizer in the `Transformers` library, we need to wrap it in a `PreTrainedTokenizerFast`. To wrap the tokenizer , we can either pass the tokenizer we built as a or pass the tokenizer file we saved. The key thing to remember is that we have to manually set all the special tokens, since that class can’t infer from the `tokenizer` object which token is the mask token, such as the `[CLS]` token.
```python
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
```

In paralell, if we are using a specific tokenizer class (such as `BertTokenizerFast`), we only need to specify the special tokens that are different from the default:
```python
from transformers import BertTokenizerFast

wrapped_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
```

### Building a BPE tokenizer from scratch
#### 1. Instantiate a `Tokenizer` object
```python
tokenizer = Tokenizer(models.BPE())
```
Because GPT-2 uses byte-level BPE, we don't need to specify an `unk_token`. Also, GPT-2 does not use a normalizer.

#### 2. Pre-tokenization
The `ByteLevel`is not to add a space at the beggining of a sentence.
```python
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

#### 3. Training
For GPT-2, the only special token is the end-of-text token:
```python
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
```

Alternatively, the tokenzier can be trianed directly on text files:
```python
tokenizer.model = models.BPE()
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
```

#### 4. Post-processing
Apply the byte-level post-processing for the GPT-2 tokenizer. The `trim_offsets=False` indicates to the post-processor that we should leaved the offsets of tokens that begin with `Ġ`, which will point to the space before the word, not the first character of the word.

```python
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
```

We can save the tokenizer, and wrap it in a `PreTrainedTokenizerFast` if we want to use it in the `Transformers` library:
```python
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)
```

Alternatively, we can wrap the tokenizer in `GPT2TokenizerFast`:
```python
from transformers import GPT2TokenizerFast

wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
```