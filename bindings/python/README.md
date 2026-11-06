# tokenizers Python bindings

This module is still experimental, expect the API to change or break.
If used in a production setting, please pin the version number.

## Install

```
pip install --pre tokenizers
```


## Usage

```python
from tokenizers import Padding, Tokenizer

# A `Tokenizer` represents a tokenizer's spec (json) and can be edited.
# You can load it from the JSON file
tokenizer = Tokenizer.from_file("tokenizer.json")
# Or load the `tokenizer.json` of a model repo on the Hub.
# Downloads go through the `huggingface_hub` package
tokenizer = Tokenizer.from_pretrained("openai-community/gpt2", revision="main")

# `compile()` compiles the specification to an optimized tokenizer in rust.
# The resulting `CompiledTokenizer` object can encode text to token ids and decode ids back to text.
# It's immutable and can be shared among threads.
compiled = tokenizer.compile()

encoding = compiled.encode("Hello there, how are you?")
encoding.ids             # [101, 7592, 2045, 1010, 2129, 2024, 2017, 1029, 102]
encoding.type_ids        # [0, 0, 0, 0, 0, 0, 0, 0, 0]
encoding.attention_mask  # [1, 1, 1, 1, 1, 1, 1, 1, 1]

# The same fields as numpy arrays. Each read returns a *new copy* that the caller owns.
encoding.ids_array             # array([101, 7592, 2045, 1010, 2129, 2024, 2017, 1029, 102], dtype=uint32)
encoding.type_ids_array        # array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint32)
encoding.attention_mask_array  # array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=uint32)
torch.from_numpy(encoding.ids_array)  # no-copy translation

compiled.encode("Hello", add_special_tokens=False).ids  # no [CLS]/[SEP]
compiled.decode(encoding.ids)                           # "hello there, how are you?"
compiled.decode(encoding.ids_array)                     # a numpy array works too
compiled.decode(encoding.ids, skip_special_tokens=False)

# Pad one call. `length=None` pads each batch to its longest item.
short, long = compiled.encode_batch(
    ["Hello", "Hello there, how are you?"],
    padding=Padding(direction="left", pad_id=0, pad_token="[PAD]"),
)
len(short) == len(long)  # True
short.attention_mask     # [0, 0, 0, 0, 0, 0, 1, 1, 1]
compiled.encode("Hello", padding=Padding(length=16))  # exactly 16 ids

# Padding and truncation can also be set on the spec.
# `padding=None` switches the default off for one call.
tokenizer.padding = Padding(pad_to_multiple_of=8)
padded = tokenizer.compile()
padded.encode("Hello", padding=None)
```

`examples/` holds a runnable script for each of these; `make examples` runs them all.

## multiprocessing and Worker processes

Every class here can be pickled, so a tokenizer can be sent to a worker process and its encodings
sent back: `multiprocessing`, a PyTorch `DataLoader`, `datasets.map(num_proc=...)`.

A pickled `CompiledTokenizer` carries its whole `tokenizer.json`, and unpickling compiles it again.
With the `"spawn"` start method, we recommend compiling the tokenizer once per worker to avoid unpickle compilation overhead.

```python
def compile_tokenizer(tokenizer):
    global compiled
    compiled = tokenizer.compile()


def encode(text):
    return compiled.encode(text)


with multiprocessing.get_context("spawn").Pool(4, initializer=compile_tokenizer, initargs=(tokenizer,)) as pool:
    encoded = list(pool.imap(encode, texts, chunksize=32))
```

With the `"fork"` start method, compiling before starting the pool is recommended: the workers inherit the compiled tokenizer
and nothing is pickled.

## Developers

- `make develop` regenerates the `.pyi` stub, rebuilds the extension (debug) and reinstalls it
  in editable mode.
- `make test` does the above, then runs `tests/` with pytest.
- `make examples` does the above, then runs every script in `examples/`.
- `make stubs` regenerates `python/tokenizers/tokenizers.pyi`, including the members
  `python/tokenizers/__init__.py` attaches to the Rust classes.
- `make style` / `make check-style` regenerate the stub, then format/lint the Rust and
  Python sides and type-check with `ty`. CI runs `make check-style` and fails if the
  regenerated stub differs from the committed one.

### Training

Train a tokenizer from UTF-8 text files or an iterable of strings:

```python
from tokenizers import Tokenizer, models, decoders, trainers

trainer = trainers.MinGramTrainer(vocab_size=8000, show_progress=False)
tokenizer = Tokenizer(models.Unigram([]))
tokenizer.decoder = decoders.ByteFallback()
tokenizer.train_from_iterator((line for line in ["some text", "more text"]), trainer=trainer)
tokenizer.save("tokenizer.json")

# Convenience form: create a tokenizer directly from files.
tokenizer = trainer.train(["corpus.txt"])
```

`BpeTrainer`, `UnigramTrainer`, and `MinGramTrainer` are available in
`tokenizers.trainers`. Unigram-family trainers accept `seed_init="substrings"`
or `"bpe"`; MinGram also accepts `prune_criterion="minimum_increase"`.
Training a tokenizer loaded from a file preserves its preprocessing and
post-processing configuration, and trains on the same normalized spans used
at encoding time. Iterator training currently collects the corpus in memory.

Use `alphabet_mode="byte_level"` with any trainer to train
on the 243 mapped byte symbols that occur in valid UTF-8, with the corresponding normalizer
and decoder installed on a newly created tokenizer. Bytes C0, C1 and F5–FF
are excluded; the general byte mapping and existing models remain unchanged.
`"legacy"` retains the
original Unigram character alphabet; `"utf8"` selects the UTF-8 fallback
inventory. `initial_alphabet` accepts additional required single characters.

To reserve a fixed digit inventory, pass `fixed_digits="single"` (10 digits),
`"single_marked"` (40 boundary forms), or `"triples"` (1,110 decimal
strings of one to three digits, grouped from the right). BPE rejects fixed-digit
training. The matching splitter
is installed in newly created tokenizers. Configured pipelines must already contain the matching fixed-digit splitter
in a safe order. The trainer infers its inventory and verifies explicit overrides.

Configure the tokenizer's pipeline separately from the trainer:

```python
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

tokenizer = Tokenizer(models.Unigram([], byte_fallback=True))
tokenizer.normalizer = normalizers.Sequence([
    normalizers.ClaudeNormalizer(fold_quotes=True),
    normalizers.MarkBoundaries(allcaps_min=4, marks_part_of_words=True, digit_group_size=3),
])
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.MarkSplit(),
    pre_tokenizers.FixedDigits(group_size=3),
])
tokenizer.decoder = decoders.Unmark()

tokenizer.train_from_iterator(
    ["Hello น้ำ नमस्ते 1234567"] * 100,
    trainer=trainers.MinGramTrainer(vocab_size=8000, alphabet_mode="utf8"),
)
tokenizer.save("claude-fixed.json")
```

The trainer reserves the inventory specified by `FixedDigits`, so the policy
does not need to be repeated in the trainer. Conflicting trainer options fail before training. A configured `FixedDigits`
splitter must be last, with only `MarkSplit` allowed before it.
Training preserves the configured normalizer, splitter and decoder.

An optional factory constructs those same public components:

```python
from tokenizers import create_claude_tokenizer, trainers

tokenizer = create_claude_tokenizer(digits="triples", preset="fixed")
tokenizer.train(["corpus.txt"], trainer=trainers.MinGramTrainer(
    vocab_size=8000, alphabet_mode="utf8",
))
```

The factory requires a digit choice: `"single"`, `"single_marked"`, `"triples"`,
or `None` for learned digits. Presets `"fixed"`, `"v3"`, and `"v47"` select the
corrected or measured Claude marking rules. The trainer has no `recipe` option.
Use `UnigramTrainer(alphabet_mode="utf8")` for plain Unigram with the same
pipeline. The Claude marked stream currently uses UTF-8 character/fallback
mode; byte-alphabet pipelines remain separate.

Components are editable shared configuration objects. Call `tokenizer.compile()`
to make an immutable tokenizer for encoding and decoding. Existing tokenizer JSON files still load.
The exposed constructors cover BPE/Unigram, common normalizers and splitters,
and the decoders used here; they do not implement every historical HF method.
In this fork, byte mapping is a `normalizers.ByteLevel()` component:

```python
tokenizer = Tokenizer(models.Unigram([]))
tokenizer.normalizer = normalizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()
tokenizer.train_from_iterator(["hello world"], trainer=trainers.UnigramTrainer(
    vocab_size=8000, alphabet_mode="byte_level",
))
```
