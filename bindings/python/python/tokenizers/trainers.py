"""Train tokenizers on raw text using BPE, Unigram, or MinGram."""

from pathlib import Path

from .tokenizers import train as _train

_DEFAULT = object()


class _Trainer:
    kind = ""

    def __init__(
        self,
        vocab_size=8000,
        show_progress=True,
        special_tokens=None,
        *,
        seed_init=None,
        seed_size=None,
        max_piece_length=16,
        n_sub_iterations=2,
        shrinking_factor=0.75,
        seed_factor=1.1,
        prune_criterion="usage_count",
        pruning_shrinking_factor=0.0,
        skip_substring_in_batch=True,
        unk_token=None,
        min_frequency=0,
        initial_alphabet=None,
        alphabet_mode=None,
        fixed_digits=_DEFAULT,
        recipe=None,
    ):
        self.vocab_size, self.show_progress = vocab_size, show_progress
        self.special_tokens = list(special_tokens or [])
        self.initial_alphabet = list(initial_alphabet or [])
        self.alphabet_mode = alphabet_mode or ("utf8" if recipe or self.kind == "mingram" else "legacy")
        self.fixed_digits = (
            ("single" if recipe == "claude_fixed" else None) if fixed_digits is _DEFAULT else fixed_digits
        )
        self.recipe = recipe
        self.seed_init = seed_init or ("bpe" if self.kind == "mingram" else "substrings")
        self.seed_size, self.max_piece_length = seed_size, max_piece_length
        self.n_sub_iterations, self.shrinking_factor = n_sub_iterations, shrinking_factor
        self.seed_factor, self.prune_criterion = seed_factor, prune_criterion
        self.pruning_shrinking_factor = pruning_shrinking_factor
        self.skip_substring_in_batch, self.unk_token, self.min_frequency = (
            skip_substring_in_batch,
            unk_token,
            min_frequency,
        )

    def train_from_iterator(self, iterator, length=None):
        """Train on an iterable of strings and return a new Tokenizer."""
        return self._train_on(iterator, None)

    def _train_on(self, iterator, tokenizer):
        return _train(
            self.kind,
            list(iterator),
            self.vocab_size,
            self.show_progress,
            self.special_tokens,
            self.seed_init,
            self.seed_size,
            self.max_piece_length,
            self.n_sub_iterations,
            self.shrinking_factor,
            self.seed_factor,
            self.prune_criterion,
            self.pruning_shrinking_factor,
            self.skip_substring_in_batch,
            self.unk_token,
            self.min_frequency,
            self.initial_alphabet,
            self.alphabet_mode,
            self.fixed_digits,
            self.recipe,
            tokenizer,
        )

    def train(self, files):
        """Train on UTF-8 files, reading one line as each training sequence."""
        return self._train_on_files(files, None)

    def _train_on_files(self, files, tokenizer):
        def lines():
            for path in files:
                with Path(path).open(encoding="utf-8") as source:
                    yield from source

        return self._train_on(lines(), tokenizer)


class BpeTrainer(_Trainer):
    kind = "bpe"

    def __init__(self, vocab_size=30000, **kwargs):
        super().__init__(vocab_size=vocab_size, **kwargs)


class UnigramTrainer(_Trainer):
    kind = "unigram"


class MinGramTrainer(_Trainer):
    kind = "mingram"
