"""Model components for explicit tokenizer construction."""

from ._components import Model


class Unigram(Model):
    def __init__(self, vocab=None, unk_id=None, byte_fallback=False, *, objective="max_score"):
        super().__init__(
            {
                "type": "Unigram",
                "vocab": [list(piece) for piece in (vocab or [])],
                "unk_id": unk_id,
                "byte_fallback": byte_fallback,
                "objective": objective,
            }
        )


class BPE(Model):
    def __init__(
        self,
        vocab=None,
        merges=None,
        *,
        unk_token=None,
        dropout=None,
        continuing_subword_prefix=None,
        end_of_word_suffix=None,
        fuse_unk=False,
        byte_fallback=False,
        ignore_merges=False,
        byte_level=False,
    ):
        super().__init__(
            {
                "type": "BPE",
                "vocab": dict(vocab or {}),
                "merges": [list(pair) for pair in (merges or [])],
                "unk_token": unk_token,
                "dropout": dropout,
                "continuing_subword_prefix": continuing_subword_prefix,
                "end_of_word_suffix": end_of_word_suffix,
                "fuse_unk": fuse_unk,
                "byte_fallback": byte_fallback,
                "ignore_merges": ignore_merges,
                "byte_level": byte_level,
            }
        )


_TYPES = {"Unigram": Unigram, "BPE": BPE}
