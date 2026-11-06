"""Optional factories that assemble explicit pipeline components."""

from . import decoders, models, normalizers, pre_tokenizers
from ._components import digit_policy
from .tokenizers import Tokenizer


def create_claude_tokenizer(*, digits, preset="fixed"):
    """Build a Claude-inspired pipeline; choose digits explicitly, or None to learn them.

    Train with MinGramTrainer or UnigramTrainer using alphabet_mode="utf8".
    The factory only configures the pipeline. It does not choose a trainer.
    """
    if preset not in ("fixed", "v3", "v47"):
        raise ValueError("preset must be fixed, v3, or v47")
    policy = None if digits is None else pre_tokenizers.FixedDigits(*digit_policy(digits))
    marking = {
        "fixed": normalizers.MarkBoundaries,
        "v3": normalizers.MarkBoundaries.v3,
        "v47": normalizers.MarkBoundaries.v47,
    }[preset](fixed_digits=policy)
    tokenizer = Tokenizer(models.Unigram(byte_fallback=True))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.ClaudeNormalizer(fold_quotes=preset != "v47"),
            marking,
        ]
    )
    splitters = [pre_tokenizers.MarkSplit()]
    if policy is not None:
        splitters.append(policy)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(splitters)
    tokenizer.decoder = decoders.Unmark()
    return tokenizer
