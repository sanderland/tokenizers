# CTok / MinGram review

Reviewed on 2026-11-06. Branch `ctok-mingram`, based on
`hackathon/mingram-trainer` at `2bfa100c`. Reference checkouts are
`sanderland/ctok` at `8c43ffd` and `sanderland/script_tok` at `7e7dc03`.

The core MinGram encoder works, and the larger UTF-8 compression check shows
a small held-out gain over BPE. The first pass fixed vocabulary accounting,
unsafe configuration values, stale Unicode categories, and fallback pricing.
There are still differences from the research implementation and practical
API gaps to resolve before presenting this as a finished user-facing trainer.

## What the Claude-inspired path does

The example composes `ClaudeNormalizer`, `MarkBoundaries`, `MarkSplit`, a
MinGram-trained `Unigram`, and `Unmark`. Training and inference use the same
cleanup, marked stream, and spans.

Cleanup applies NFC, removes selected controls, changes NUL to a space,
folds selected space separators, strips BMP private-use input, and composes
Thai SARA AM. Quote folding is enabled for the v3-style default and disabled
for the v4.7 preset. Cleanup is lossy; decoding recovers the cleaned text.

Word spans receive beginning/end markers. Title case receives a shift code.
The default keeps combining marks inside words and uses the v3 caps code for
all-uppercase spans of at least four characters. The `v3()` and `v47()` presets
retain the measured mark-splitting rules; v4.7 disables the caps code.
Variation selectors ride with their bases. CJK, Hangul syllables, and astral
characters follow the markerless path. There is no SCRIPT encoding or
script-boundary restriction in this implementation.

`MarkSplit` cuts before beginning markers, after end markers, and at whitespace
transitions. Case codes stay with their words. A single space represented by
an end/begin marker seam is restored by `Unmark`. Markerless regions can
contain several kinds of character; this is deliberately a coarser policy
than a GPT-style regular expression.

This splitter is an additional restriction relative to ctok's whole-stream
tiler. The presets reproduce boundary rewrites, with chat framing removed;
they do not import Claude's measured vocabulary or reproduce message counts.
Offsets refer to the rewritten stream rather than original input bytes.

### Digits

There is no one-digit, left-to-right-three-digit, or right-to-left-three-digit
rule. Examples below show v4.7 frameless spans; `B` and `E` stand for the
beginning/end markers.

| Input | Pretokens |
| --- | --- |
| `1234567890` | `1234567890` |
| `123.456` | `123.456` |
| `12 34` | `12`, space, `34` |
| `a123b` | `BaE`, `123`, `BbE` |
| `١٢٣ ٤٥٦` | `١٢٣E`, `B٤٥٦` |
| `१२३ ४५६` | `१२३E`, `B४५६` |
| `１２３ ４５６` | `１２３E`, `B４５６` |

Vocabulary membership decides the eventual number of tokens inside these
spans. Long numbers and decimal-like strings may become single learned
pieces. Preserve this policy for the Claude-style comparison; make an
arithmetic-oriented digit policy an explicit pipeline choice if desired.

## Defaults and fixes

The retained defaults are BPE initialization, two hard-EM sub-iterations per
round, usage-count pruning, one shrink step, and a 16-character maximum piece
length. If pruning is needed, another round re-estimates scores afterward.
Minimum-increase pruning remains an advanced option. The encoding objective
compares token counts exactly, then log-probabilities, then a deterministic
positional tie rule. It does not expose an unsafe score/count tradeoff knob.

The new default seed overshoots the learnable multi-character budget by 10%.
The previous unset cap was one million pieces, inherited from standard
Unigram, which changes both training cost and the intended MinGram recipe.
`seed_factor` exposes this directly without asking callers to calculate
alphabet/fallback reservations. An explicit `seed_size` overrides it. The
paper's highlighted MinGram setting uses factor 1.15; the current script_tok
configuration defaults to 1.1, and both are included in the larger check.

The UTF-8 safety rules are fixed. Seeding begins from complete characters,
so learned pieces cannot attach half a character to another character.
Fallback entries are valid non-ASCII bytes plus proper UTF-8 prefixes of
observed characters. Those prefixes begin at the leading byte. No option
allows continuation-only merges. This achieves the safety invariant by
construction; it does not reproduce script_tok's byte-level merge history.

Changes made in this pass:

1. Reserve byte fallback and special-token slots before EM and pruning.
   The required alphabet is present throughout training. Finalization no
   longer silently charges these pieces against a previously pruned vocab.
2. Fix an exact-budget finalizer overflow. A request for 249 pieces could
   emit 251 on a tiny ASCII corpus. A full budget is now checked before
   inserting another learned piece.
3. Reject zero EM iterations, zero lengths/seeds, empty positive-count
   corpora, invalid shrinking factors, insufficient seed alphabets, duplicate
   specials, and specials colliding with required characters or byte pieces.
   Checks run at execution time, including after serde/public-field changes.
4. Use the shared current Unicode classifier for letters, marks, digits,
   punctuation, and symbols. Stale categories split newer combining marks
   and misclassified newer letters, including Georgian Mtavruli.
5. Price byte fallback emissions inside minimum-token inference. An unknown
   emoji costs four byte tokens when no prefix helps, even though it is one
   lattice node. A regression test covers an overlapping-vocabulary case
   where the old encoder chose six emitted tokens over five.
6. Include required singles in gradual-pruning target calculations.
   Minimum-increase trimming now recomputes batches until it reaches the
   requested size, including with substring skipping enabled.
7. Report frequency-weighted training token counts using `u64`; stabilize
   corpus ordering and aggregate duplicate sentence weights during BPE seeding.
8. Restore missing feature guards on the JSON float writer and its helper.
   The inherited branch could not compile a BPE-only serialization build.

Vocabulary size remains a cap. Hard EM can discard unused pieces and return
fewer entries. All observed characters, all ASCII, and five boundary markers
are required. Small multilingual budgets can therefore fail early.

## Checks and compression experiments

The Rust encoder, serialization, and trainer suites pass, including the
pinned GPT-2 integration fixture. The selected script_tok UTF-8, merge-policy,
and MinGram tests pass, 54 tests. The existing Claude-style demo round-trips
all 12 examples and its fixed mark policy uses fewer tokens on 10.

The direct audit compares 13,880 frameless rewrites with ctok, with no
differences, and checks 20,820 marking/decoding cases with no round-trip
failures. It covers randomized mixed text, every BMP mark in Python's Unicode
15.1 database, sampled BMP letters/numbers/punctuation/symbols, and explicit
edge cases. This is regression coverage rather than an exhaustive proof for
all Unicode strings or an Anthropic API validation.

On an identical fixed UTF-8 vocabulary, token counts agree with script_tok on
all 500 randomized cases. A controlled fixed-vocabulary hard-EM fixture also
gives the same corpus token total, 18. Rust floors unused scores at 0.5;
script_tok's raw training object can retain negative infinity, so this does
not establish identical probabilities or complete training parity.

### Rosetta Code holdout

Train on ctok's existing 1,741-document fixture, 1,344,598 UTF-8 bytes; evaluate
on its separate 250-document holdout, 200,583 bytes. Document IDs are disjoint.
Use NFC, GPT-4o regex spans, UTF-8, and a 4,096-entry cap. All returned
vocabularies in this experiment have exactly 4,096 entries. No Claude marking
or SCRIPT representation is used in this algorithm comparison.

| Method | Held-out tokens | Change from BPE |
| --- | ---: | ---: |
| script_tok BPE | 79,722 | baseline |
| script_tok MinGram, factor 1.1 | 79,045 | -0.85% |
| Rust MinGram, factor 1.1 | 79,542 | -0.23% |
| script_tok MinGram, factor 1.15 | 79,220 | -0.63% |
| Rust MinGram, factor 1.15 | 79,594 | -0.16% |
| script_tok MinGram, factor 2 | 79,669 | -0.07% |
| Rust MinGram, factor 2 | 80,014 | +0.37% |

This reproduces the direction of a small compression gain at a small seed
overshoot. It leaves a gap to the reference and shows that a larger seed is
not automatically better. Character seeding, reserved alphabets, score
smoothing, seed-budget definitions, and maximum piece lengths differ, so the
table does not isolate the cause of that gap. Timings in the JSON use a Rust
debug build and include subprocess overhead; they are not a speed benchmark.

### Small natural-language check

Six UDHR translations use a 1,024-entry cap. Every fifth nonempty line is
held out. These corpora have only about 8–16 KB of training text per language,
and many resulting vocabularies underfill the cap.

| Language | BPE tokens | Rust MinGram, factor 1.1 | Change |
| --- | ---: | ---: | ---: |
| English | 641 | 647 | +0.94% |
| German | 738 | 746 | +1.08% |
| Finnish | 740 | 722 | -2.43% |
| Russian | 758 | 766 | +1.06% |
| Arabic | 617 | 642 | +4.05% |
| Korean | 566 | 567 | +0.18% |

This tiny check does not reproduce a general compression improvement.
Factor 2 performs worse than 1.1 in both implementations on every language.
Evaluate reference models after save/load, as the research scripts do:
script_tok's loader floors unused atomic scores to -100. Freshly trained
objects can yield different held-out paths. The JSON records both counts.

## Remaining work worth doing

- **Keep plain Unigram compatibility separate.** The inherited branch uses
  shared seeding/finalization helpers to force MinGram's ASCII/marker alphabet
  and byte floor into `UnigramTrainer` as well. That changes its small-vocab
  behavior and fallback policy. Isolate that policy before an upstream PR.
- **Close the UTF-8 training gap.** The reference learns partial prefixes and
  characters through constrained byte merges. The Rust version protects every
  observed character and installs every observed prefix afterward. The
  Rosetta gap warrants a controlled seed/vocab comparison; larger CJK corpora
  also need a principled rare-character coverage policy within the cap.
- **Expose a practical training entry point.** The Python binding explicitly
  leaves trainers unbound. Rust works, but the example has a hardcoded tiny
  corpus. A file/iterator training interface with matching preprocessing is
  needed for the intended "train your own" workflow.
- **Treat presets and formats carefully.** Pin Unicode data for historical
  Claude presets, distinguish rewritten offsets from source offsets, and
  document that custom normalizers/models are fork-specific JSON components.
  The generic decoder's `<0x...>` spelling ambiguity also remains for
  specially constructed raw vocabularies.

The next compression study should use a bounded FineWeb/FineWeb-2 UTF-8
sample, a 32K vocabulary budget, and held-out Goldfish evaluation, with
actual alphabets and sizes recorded. The local fixture experiments above
do not reproduce the paper's full training scale, morphology evaluation,
or downstream language-model results.

## Reproduce this review

From the repository root, build the probe:

```sh
cd tokenizers
cargo build -p tk-train --example review_mingram --features normalizers
cd ..
python scripts/review_mingram.py --output docs/experiments/ctok-mingram-parity.json
python scripts/compare_mingram_compression.py --output /tmp/udhr-compression.json
python scripts/compare_mingram_compression.py --corpus rosetta --vocab-size 4096 \
  --factors 1.1 1.15 2.0 --output /tmp/rosetta-compression.json
```

The Python scripts import the sibling `ctok` and `script_tok` checkouts and
need script_tok's runtime dependencies plus `regex`. They read the committed
fixtures and do not download corpora. Reference BPE training uses one local
worker process. Results, fixture hashes, settings, sizes, and timing caveats
are under [experiments](experiments/).

Research context: [MinGram](https://arxiv.org/html/2606.27019) describes the
minimum-path training and pruning methods. [Explicit Boundary Markers for
Subword Vocabularies](https://arxiv.org/html/2608.08847) studies the boundary
and case representation. That paper motivates the representation through
reduced duplication and downstream quality; it does not claim a general
compression improvement from boundary markers alone.
