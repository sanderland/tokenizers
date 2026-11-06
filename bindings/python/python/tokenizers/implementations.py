"""Optional factories that assemble explicit pipeline components."""

from . import decoders, models, normalizers, pre_tokenizers
from .tokenizers import Tokenizer


def create_claude_tokenizer(*, digits, preset="fixed"):
    """Build a Claude-inspired pipeline; choose digits explicitly, or None to learn them.

    Train with MinGramTrainer or UnigramTrainer using alphabet_mode="utf8".
    The factory only configures the pipeline. It does not choose a trainer.
    """
    if preset not in ("fixed", "v3", "v47"):
        raise ValueError("preset must be fixed, v3, or v47")
    values = {"single": (1, False), "single_marked": (1, True), "triples": (3, False)}
    if digits is not None and digits not in values:
        raise ValueError("digits must be single, single_marked, triples, or None")
    width, borders = values.get(digits, (None, False))
    policy = None if width is None else pre_tokenizers.FixedDigits(width, borders)
    marking = normalizers.MarkBoundaries(
        allcaps_min=None if preset == "v47" else 4,
        marks_part_of_words=preset == "fixed",
        digit_group_size=width,
        digit_boundary_markers=borders,
    )
    tokenizer = Tokenizer(models.Unigram([], byte_fallback=True))
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
