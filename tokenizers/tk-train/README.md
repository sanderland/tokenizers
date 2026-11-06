# tk-train

Training half of the 🤗 Tokenizers library.

This crate builds on top of [`tk_encode`] and provides everything related to
*training* a tokenizer: the [`Trainer`] trait, every concrete `*Trainer`, the
[`TrainerWrapper`] enum and the [`Trainable`] extension (the `get_trainer`
association that used to live on `tk_encode::Model`).

There is no tokenizer-level `train` / `train_from_files` entry point: it was an
extension trait on `tk_convert`'s `TokenizerImpl`, and nothing in `tk-encode`
replaces that type -- the pipeline tokenizer hands out its model by shared
reference only. Drive a trainer directly instead: `feed`, then `train`.

## MinGram and Claude-inspired training

`MinGramTrainer` learns a Unigram vocabulary with minimum-token inference and
learned log-probabilities for ties. The default uses a BPE seed, two hard-EM
sub-iterations, and usage-count pruning in a single shrink step. The seed
overshoots the available multi-character budget by 10%, rather than using
Unigram's million-substring seed cap.

`vocab_size` includes special tokens, required characters, and UTF-8 fallback
pieces. It is an upper bound: EM can remove unused pieces and return fewer.
The required alphabet is all observed characters, all ASCII, and five boundary
markers. Fallbacks include valid non-ASCII UTF-8 bytes and proper prefixes of
observed characters. These slots are reserved before training and pruning.
A budget below this floor fails before seeding.

The BPE seed starts with whole Unicode characters. It cannot create tokens
joining half a character to its neighbor. The fallback path is always present
and its multi-byte pieces are prefixes starting at a character's leading byte.
Neither invariant has an unsafe toggle.

For the Claude-inspired pipeline, use `ClaudeNormalizer`, then
`MarkBoundaries::default()`, then `MarkSplit`, with the `Unmark` decoder.
Normalize and split training text exactly as inference does before passing
the resulting spans to the trainer. The default boundary rule keeps combining
marks inside words. Explicit `v3()` and `v47()` presets retain the measured
Claude mark policy for comparisons. See the runnable save/load demonstration:

```sh
cargo run -p tk-train --example claude_style --features normalizers
```

Useful overrides are `vocab_size`, `seed_factor`, `max_piece_length`, and
`special_tokens`. The seed cap includes the required characters and must cover
them; an explicit `seed_size` overrides `seed_factor`. The default maximum
piece length is 16 Unicode scalars. Keep EM and
pruning defaults initially; `MinimumIncrease` pruning and gradual shrinking
are advanced compression experiments. `pruning_shrinking_factor` must be in
`[0, 1)`, with zero selecting a single shrink step.

This pipeline trains a new vocabulary. It does not reconstruct Anthropic's
vocabulary, chat framing, or message token counts. ASCII digit runs are not
limited to three digits: `1234567890` is one pretoken, and even `123.456` may
be learned together. Add an explicit digit pre-tokenizer if your application
requires a different policy, and use it for both training and inference.

The Python bindings currently do not expose trainers. This workflow uses Rust.
The branch review, UTF-8 comparison scripts, and small compression experiment
are documented in [the review](../../docs/ctok-mingram-review.md).

License: Apache-2.0
