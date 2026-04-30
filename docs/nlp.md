# :hugs: Natural Language Processing

## :pushpin: Fine-tuning a masked language model
We can simply take a pretrained Transformer model from the Hugging Face Hub and fine-tune it directly to our data for a specific task. If the corpus used for pretraining is not too different from the corpus used for fine-tuning, transfer learning is likely to produce good results.

The fine-tuning of a pretrained language model on in-domain data is called ***domain adaptation***.

### Selecting a pretrained model for masked language modeling
The pretrained models for masked language modeling can be find by applying the "Fill-Mask" filter on the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=fill-mask&sort=downloads).

BERT and RoBERTa are the family of models most downloaded. In the example below, we'll use [DistilBERT](https://huggingface.co/distilbert-base-uncased), which can be trained much faster. Because DistilBERT was trained using [***knowledge distillation***](https://en.wikipedia.org/wiki/Knowledge_distillation), where a large model (e.g., BERT) is used to guide the straining of a small model that has fewer parameters. In the case of DistilBERT, it is approximatelly two times smaller than BERT.

The number of parameters can be verified using `num_parameters()`.

DistilBERT can be loaded using `AutoModelForMaskedLM`:
```python
from transformers import AutoModelForMaskedLM

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

The predictions of pretrained models depend on the corpus the model was trained on, since it learns to pick the statistical patterns present in the dataset. As BERT, DistilBERT was pre-trained on the [English Wikipedia](https://huggingface.co/datasets/wikipedia) and [BookCorpus](https://huggingface.co/datasets/bookcorpus), therefore we expect the predictions for `[MASK]` to reflect these domains.

In the example below, we need to load the tokenizer from DistilBERT to produce the inputs for the model and predict the `[MASK]`.

```python
text = "This is a great [MASK]."
```

The tokenizer can be loaded using `AutoTokenizer`T
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

Now, we can tokenize the example and pass to the model, and then extract the logits and print out the predictions.
```python
import torch

inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'{text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
```

```python
'This is a great deal.'
'This is a great success.'
'This is a great adventure.'
'This is a great idea.'
'This is a great feat.'
```

The output shows that the model predictions refer to everydat terms. Next, we are going to change the corpus domain to movie reviews.

### Dataset
We'll use the [Large Movie Review Dataset](https://huggingface.co/datasets/imdb), which is a corpus of movie reviews. By fine-tuning DistilBERT on this corpus, we expect the model to adapt its vocabulary from Wikipedia and BookCorpus that it was pretrained on to the words of moview reviews.

We can load the dataset using `load_dataset()`:
```python
from datasets import load_dataset

imdb_dataset = load_dataset("imdb")
imdb_dataset
```

```python
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
})
```

The `train` and `test` splits consists each of 25,000 reviews, and an `unsupervised` split that contains 50,000 reviews.

We can create a random sample to take a look at the dataset using `Dataset.shuffle()` and `Dataset.select()`.
```python
sample = imdb_dataset["train"].shuffle(seed=42).select(range(1))

for row in sample:
    print(f"\n'Review: {row['text']}'")
    print(f"'Label: {row['label']}'")
```

```python
'Review: This is your typical Priyadarshan movie--a bunch of loony characters out on some silly mission. His signature climax has the entire cast of the film coming together and fighting each other in some crazy moshpit over hidden money. Whether it is a winning lottery ticket in Malamaal Weekly, black money in Hera Pheri, "kodokoo" in Phir Hera Pheri, etc., etc., the director is becoming ridiculously predictable. Don\'t get me wrong; as clichéd and preposterous his movies may be, I usually end up enjoying the comedy. However, in most his previous movies there has actually been some good humor, (Hungama and Hera Pheri being noteworthy ones). Now, the hilarity of his films is fading as he is using the same formula over and over again.<br /><br />Songs are good. Tanushree Datta looks awesome. Rajpal Yadav is irritating, and Tusshar is not a whole lot better. Kunal Khemu is OK, and Sharman Joshi is the best.'
'Label: 0'
```

### Dataset processing
For both auto-regressive and masked language modelling, a common preprocessing step is to concatenate all the examples and split them into chuncks of equal size, instead of simply tokenizing individual examples, which prevents truncation of long examples and lost of information.

#### Tokenization
Therefore, we start with tokenizing the corpus without `truncation=True`. In addition, we extract the word IDs with they are available to do the word masking. Also, we remove the `text` and `label` columns.
```python
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# Use batched=True to activate fast multithreading
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
tokenized_datasets
```

```python
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 50000
    })
})
```

The model encoded the texts into `input_ids` and `attention_mask`.

#### Splitting
Now that we tokenized the dataset, the next step is to group then all and split into chunks. The size of the chunks is determined by the ammout of GPU memory. A good starting point is the model's maximum context size, which in this case is 512.

To do so, we iterate over the features and use a list comprehension to create slices of each feature, resulting in a dictionary of chunks for each feature. Because the last chunk can be smaller than the maximum chunk size, we can either drop or pad until its length equals the chunk size. Here, we'll drop the last chunk.

```python
def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result
```
:bulb: Note that in the last step of group_texts() we create a new labels column which is a copy of the input_ids one. As we’ll see shortly, that’s because in masked language modeling the objective is to predict randomly masked tokens in the input batch, and by creating a labels column we provide the ground truth for our language model to learn from.

After, we can apply `group_texts()` to the tokenized dataset using `Dataset.map()`.
```python
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
lm_datasets
```

```python
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 61289
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 59905
    })
    unsupervised: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 122963
    })
})
```

:bulb: Note that rouping and then chunking the texts has produced many more examples than our original 25,000 for the `train` and `test` splits. That's because we now have examples involving ***contiguous tokens*** that span across multiple examples from the original corpus. Because we didn't inserted `[MASK]`tokens at random positions, `input_ids` and `labels` are exactly the same.

#### Insert `[MASK]` tokens at random positions using `DataCollatorForLanguageModeling`
To randomly mask some tokens in each batch of input of texts, we have to pass to `DataCollatorForLanguageModeling` the tokenizer and the `mlm_probability` argument, which specifies what fraction of the tokens to mask. Usually, 15% is the percentage used for BERT.

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```

We can see how the random masking is working by feeding examples to the data collator.
```python
samples = [lm_datasets["train"][i] for i in range(1)]
for sample in samples:
    _ = sample.pop("word_ids")

for chunk in data_collator(samples)["input_ids"]:
    print(f"\n{tokenizer.decode(chunk)}'")
```

```python
'[CLS] bromwell [MASK] is a cartoon comedy. it ran at the same [MASK] as some other [MASK] about school life, [MASK] as " teachers ". [MASK] [MASK] [MASK] in the teaching [MASK] lead [MASK] to believe that bromwell high\'[MASK] satire is much closer to reality than is " teachers ". the scramble [MASK] [MASK] financially, the [MASK]ful students whogn [MASK] right through [MASK] pathetic teachers\'pomp, the pettiness of the whole situation, distinction remind me of the schools i knew and their students. when i saw [MASK] episode in [MASK] a student repeatedly tried to burn down the school, [MASK] immediately recalled. [MASK]...'
```

We can see that the `[MASK]` token has been randomy inserted in the input text. During training, the model will predict the masked tokens.

:bulb: Because we're using random masking, the evaluation metric won't be deterministic using the `Trainer`. Nevertheless, using :hugs: `Accelerate`, we can use a custom evaluation loop to freeze the randomness.

:bulb: Instead of masking individual tokens, we mask whole words togheter, which is called ***whole word masking***, which requires to build a custom data collator that takes a list of samples and converts them into a batch.

To build a data collator for ***whole word masking***, we'll use the word IDs to make a map between word indices and the corresponding tokens, then randomly decided which words to mask and apply that mask on the input text.
```python
import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)
```

We can see how the random whole word masking is working by feeding examples to the data collator.
```python
samples = [lm_datasets["train"][i] for i in range(1)]
batch = whole_word_masking_data_collator(samples)

for chunk in batch["input_ids"]:
    print(f"\n{tokenizer.decode(chunk)}'")
```

```python
'[CLS] bromwell high is a cartoon comedy [MASK] it ran at the same time as some other programs about school life, such as " teachers ". my 35 years in the teaching profession lead me to believe that bromwell high\'s satire is much closer to reality than is " teachers ". the scramble to survive financially, the insightful students who can see right through their pathetic teachers\'pomp, the pettiness of the whole situation, all remind me of the schools i knew and their students. when i saw the episode in which a student repeatedly tried to burn down the school, i immediately recalled.....'
```

### Training
First, we specify the training arguments for the `Trainer`:
```python
from transformers import TrainingArguments

batch_size = 64
# Show the training loss with every epoch
logging_steps = len(lm_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    fp16=True,
    logging_steps=logging_steps,
)
```

:bulb: By default, the `Trainer` will remove any columns that are not part of the model’s `forward()` method. This means that if you’re using the whole word masking collator, you’ll also need to set `remove_unused_columns=False` to ensure we don’t lose the `word_ids` column during training.

Next, we instantiate the `Trainer`:
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=whole_word_masking_data_collator, # We can use the standard data_collator
    tokenizer=tokenizer,
)
```

Then, we're ready to train the model:
```python
trainer.train()
```

### Evaluation
A good language model is one that assigns high probabilities to sentences that are grammatically correct, and low probabilities to nonsense sentences.

One way to measure the quality of our language model is to calculate the probabilities it assigns to the next word in all the sentences of the test set. High probabilities indicates that the model is not “surprised” or “perplexed” by the unseen examples, and suggests it has learned the basic patterns of grammar in the language.

There are various mathematical definitions of perplexity, but the one we’ll use defines it as the exponential of the cross-entropy loss. Thus, we can calculate the perplexity of our pretrained model by using the Trainer.evaluate() function to compute the cross-entropy loss on the test set and then taking the exponential of the result. A lower perplexity score means a better language model

```python
import math

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
```

```python
Perplexity: 11.32
```

### Accelerate
`DataCollatorForLanguageModeling` also applies random masking with each evaluation, so we’ll see some fluctuations in our perplexity scores with each training run. One way to eliminate this source of randomness is to apply the masking once on the whole test set, and then use the default data collator in :hugs: Transformers to collect the batches during evaluation


First, we write a function that applies the masking on a batch, as `DataCollatorForLanguageModeling`.
```python
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
```

Next, we’ll apply this function to our test set and drop the unmasked columns so we can replace them with the masked ones.
:bulb: an use whole word masking by replacing the `data_collator` above with `whole_word_masking_data_collator`, in which case we should remove the first line.

```python
downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)
```

Then, we can set up the `DataLoader` using the `default_data_collator`:
```python
from torch.utils.data import DataLoader
from transformers import default_data_collator

batch_size = 64
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)
```

Now, we can follow the workflow of :hugs: Accelerate. First, we load the pretrained model:
```python
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

Then, we specify the optimizer:
```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

Now, we can write the `Accelerator` object for training:
```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

Then, we can specify the learning rate scheduler:
```python
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

Finally, we write the training and evaluation loop:
```python
from tqdm.auto import tqdm
import torch
import math

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f"Epoch {epoch}: Perplexity: {perplexity}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
```

### Inference
We can interact with your fine-tuned model either by using its widget on the Hub or locally with the pipeline from :hugs: `Transformers`. Let’s use the latter to download our model using the fill-mask pipeline:
```python
from transformers import pipeline

mask_filler = pipeline(
    "fill-mask", model="huggingface-course/distilbert-base-uncased-finetuned-imdb"
)
```

Then, we can feed the text (i.e., "This is a gread [MASK]") to see what is the model prediction.
```python
preds = mask_filler(text)

for pred in preds:
    print(f"{pred['sequence']}")
```

```python
'this is a great movie.'
```

In conclusion, the model adapted its weights to predict words associated with the movies domain.

## :pushpin: 