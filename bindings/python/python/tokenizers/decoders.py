"""Decoder components for reversing the tokenizer's text representation."""

from ._components import Decoder, members


class Sequence(Decoder):
    def __init__(self, decoders):
        super().__init__({"type": "Sequence", "decoders": members("decoder", decoders)})


class Unmark(Decoder):
    def __init__(self):
        super().__init__({"type": "Unmark"})


class ByteLevel(Decoder):
    def __init__(self):
        super().__init__({"type": "ByteLevel"})


class ByteFallback(Decoder):
    def __init__(self):
        super().__init__({"type": "ByteFallback"})


class Fuse(Decoder):
    def __init__(self):
        super().__init__({"type": "Fuse"})


_TYPES = {cls.__name__: cls for cls in (Sequence, Unmark, ByteLevel, ByteFallback, Fuse)}
