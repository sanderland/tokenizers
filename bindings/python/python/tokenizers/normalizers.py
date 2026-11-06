"""Normalizer components. Sequence order is the normalization order."""

from ._components import Normalizer, digit_policy, members


class Sequence(Normalizer):
    def __init__(self, normalizers):
        super().__init__({"type": "Sequence", "normalizers": members("normalizer", normalizers)})


class ClaudeNormalizer(Normalizer):
    def __init__(self, fold_quotes=True):
        super().__init__({"type": "ClaudeNormalizer", "fold_quotes": fold_quotes})


class MarkBoundaries(Normalizer):
    def __init__(self, allcaps_min=4, marks_part_of_words=True, *, fixed_digits=None):
        config = {"type": "MarkBoundaries", "allcaps_min": allcaps_min, "marks_part_of_words": marks_part_of_words}
        if fixed_digits is not None:
            config["digit_group_size"], config["digit_boundary_markers"] = digit_policy(fixed_digits)
        super().__init__(config)

    @classmethod
    def v3(cls, *, fixed_digits=None):
        return cls(allcaps_min=4, marks_part_of_words=False, fixed_digits=fixed_digits)

    @classmethod
    def v47(cls, *, fixed_digits=None):
        return cls(allcaps_min=None, marks_part_of_words=False, fixed_digits=fixed_digits)


class ByteLevel(Normalizer):
    def __init__(self):
        super().__init__({"type": "ByteLevel"})


class Lowercase(Normalizer):
    def __init__(self):
        super().__init__({"type": "Lowercase"})


class NFC(Normalizer):
    def __init__(self):
        super().__init__({"type": "NFC"})


class NFD(Normalizer):
    def __init__(self):
        super().__init__({"type": "NFD"})


class NFKC(Normalizer):
    def __init__(self):
        super().__init__({"type": "NFKC"})


class NFKD(Normalizer):
    def __init__(self):
        super().__init__({"type": "NFKD"})


class StripAccents(Normalizer):
    def __init__(self):
        super().__init__({"type": "StripAccents"})


class Strip(Normalizer):
    def __init__(self, left=True, right=True):
        super().__init__({"type": "Strip", "strip_left": left, "strip_right": right})


class Prepend(Normalizer):
    def __init__(self, prepend):
        super().__init__({"type": "Prepend", "prepend": prepend})


_TYPES = {
    cls.__name__: cls
    for cls in (
        Sequence,
        ClaudeNormalizer,
        MarkBoundaries,
        ByteLevel,
        Lowercase,
        NFC,
        NFD,
        NFKC,
        NFKD,
        StripAccents,
        Strip,
        Prepend,
    )
}
