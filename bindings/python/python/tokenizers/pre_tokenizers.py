"""Pre-tokenizer components. A Sequence applies each splitter in order."""

from ._components import PreTokenizer, members


class Sequence(PreTokenizer):
    def __init__(self, pretokenizers):
        super().__init__({"type": "Sequence", "pretokenizers": members("pre_tokenizer", pretokenizers)})


class MarkSplit(PreTokenizer):
    def __init__(self):
        super().__init__({"type": "MarkSplit"})


class FixedDigits(PreTokenizer):
    def __init__(self, group_size=1, boundary_markers=False):
        if group_size not in (1, 3) or not isinstance(group_size, int) or isinstance(group_size, bool):
            raise ValueError("group_size must be 1 or 3")
        if not isinstance(boundary_markers, bool) or (group_size == 3 and boundary_markers):
            raise ValueError("boundary markers are only supported with single digits")
        super().__init__({"type": "FixedDigits", "group_size": group_size, "boundary_markers": boundary_markers})

    @classmethod
    def single(cls, boundary_markers=False):
        return cls(1, boundary_markers)

    @classmethod
    def triples(cls):
        return cls(3, False)


class Digits(PreTokenizer):
    def __init__(self, individual_digits=False):
        super().__init__({"type": "Digits", "individual_digits": individual_digits})


class Whitespace(PreTokenizer):
    def __init__(self):
        super().__init__({"type": "Whitespace"})


class WhitespaceSplit(PreTokenizer):
    def __init__(self):
        super().__init__({"type": "WhitespaceSplit"})


class Split(PreTokenizer):
    def __init__(self, pattern, behavior="Isolated", invert=False):
        if not isinstance(pattern, str):
            raise TypeError("Split pattern must be a literal string")
        super().__init__({"type": "Split", "pattern": {"String": pattern}, "behavior": behavior, "invert": invert})


_TYPES = {cls.__name__: cls for cls in (Sequence, MarkSplit, FixedDigits, Digits, Whitespace, WhitespaceSplit, Split)}
