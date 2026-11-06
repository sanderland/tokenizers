"""Configuration objects shared by the explicit pipeline bindings."""

import copy
import importlib
import json


class Component:
    """Immutable component configuration; assign a new object to change a pipeline."""

    _kind = None

    def __init__(self, config):
        object.__setattr__(self, "_data", copy.deepcopy(config))

    @property
    def _config(self):
        return copy.deepcopy(self._data)

    def __getattr__(self, name):
        if name in self._data:
            return copy.deepcopy(self._data[name])
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("Component configurations are immutable; assign a new component to the tokenizer")

    def __repr__(self):
        fields = ", ".join(f"{key}={value!r}" for key, value in self._data.items() if key != "type")
        if len(fields) > 240:
            fields = fields[:237] + "..."
        return f"{self._data['type']}({fields})"

    def __eq__(self, other):
        return isinstance(other, Component) and self._kind == other._kind and self._data == other._data

    def __reduce__(self):
        return from_json, (self._kind, json.dumps(self._data, ensure_ascii=False, allow_nan=False))


class Model(Component):
    _kind = "model"


class Normalizer(Component):
    _kind = "normalizer"


class PreTokenizer(Component):
    _kind = "pre_tokenizer"


class Decoder(Component):
    _kind = "decoder"


_BASES = {"model": Model, "normalizer": Normalizer, "pre_tokenizer": PreTokenizer, "decoder": Decoder}
_MODULES = {"model": "models", "normalizer": "normalizers", "pre_tokenizer": "pre_tokenizers", "decoder": "decoders"}


def to_json(kind, component):
    if kind not in _BASES:
        raise ValueError(f"Unknown component kind {kind!r}")
    if component is None:
        if kind == "model":
            raise TypeError("model cannot be None")
        return "null"
    if not isinstance(component, _BASES[kind]):
        raise TypeError(f"Expected a {kind} component, got {type(component).__name__}")
    return json.dumps(component._config, ensure_ascii=False, allow_nan=False)


def from_json(kind, serialized):
    config = json.loads(serialized)
    if config is None:
        return None
    module = importlib.import_module(f"tokenizers.{_MODULES[kind]}")
    cls = module._TYPES.get(config["type"], _BASES[kind])
    obj = cls.__new__(cls)
    Component.__init__(obj, config)
    return obj


def members(kind, components):
    result = []
    for component in components:
        if component is None:
            raise TypeError("Sequence members cannot be None")
        result.append(json.loads(to_json(kind, component)))
    return result


def digit_policy(policy):
    if isinstance(policy, PreTokenizer) and policy._config.get("type") == "FixedDigits":
        return policy.group_size, policy.boundary_markers
    values = {"single": (1, False), "single_marked": (1, True), "triples": (3, False)}
    if policy not in values:
        raise ValueError("digits must be single, single_marked, triples, or a FixedDigits component")
    return values[policy]
