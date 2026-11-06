"""Exercise explicit pipeline configuration through training and serialization."""

import json
import pickle

import pytest

from tokenizers import Tokenizer, create_claude_tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


def test_explicit_unigram_pipeline_train_reload_and_replace(tmp_path):
    tokenizer = Tokenizer(models.Unigram(byte_fallback=True))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.ClaudeNormalizer(fold_quotes=True),
            normalizers.MarkBoundaries(marks_part_of_words=True, fixed_digits="triples"),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.MarkSplit(),
            pre_tokenizers.FixedDigits.triples(),
        ]
    )
    tokenizer.decoder = decoders.Unmark()
    assert isinstance(tokenizer.model, models.Unigram)
    assert isinstance(tokenizer.normalizer, normalizers.Sequence)
    assert isinstance(tokenizer.pre_tokenizer, pre_tokenizers.Sequence)
    assert isinstance(tokenizer.decoder, decoders.Unmark)
    corpus = ["hello น้ำ नमस्ते 1234567"] * 10
    trainer = trainers.MinGramTrainer(vocab_size=1600, alphabet_mode="utf8", show_progress=False)
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    text = "Hello น้ำ 1234567 “yes” 🦊"
    expected = 'Hello น้ำ 1234567 "yes" 🦊'
    assert tokenizer.decode(tokenizer.encode(text).ids) == expected
    assert tokenizer.decode_tokens(tokenizer.encode("1234567").ids) == ["1", "234", "567"]
    path = tmp_path / "explicit.json"
    tokenizer.save(path)
    loaded = Tokenizer.from_file(path)
    assert isinstance(loaded.normalizer, normalizers.Sequence)
    assert loaded.pre_tokenizer == tokenizer.pre_tokenizer
    assert loaded.decoder == tokenizer.decoder
    assert loaded.decode(loaded.encode(text).ids) == expected
    assert pickle.loads(pickle.dumps(loaded)).decode(loaded.encode(text).ids) == expected
    assert pickle.loads(pickle.dumps(loaded.normalizer)) == loaded.normalizer
    assert len(loaded.model.vocab) <= 1600


def test_explicit_bpe_and_byte_alphabet_unigram(tmp_path):
    for model, trainer in [
        (models.BPE(), trainers.BpeTrainer(vocab_size=270, alphabet_mode="byte_level", show_progress=False)),
        (models.Unigram(), trainers.UnigramTrainer(vocab_size=270, alphabet_mode="byte_level", show_progress=False)),
    ]:
        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(["hello world"] * 3, trainer=trainer)
        assert tokenizer.decode(tokenizer.encode("unseen 🦊").ids) == "unseen 🦊"
        path = tmp_path / "byte.json"
        tokenizer.save(path)
        assert Tokenizer.from_file(path).decode(tokenizer.encode("unseen 🦊").ids) == "unseen 🦊"


def test_component_setters_are_atomic_and_validate_slot(tmp_path):
    tokenizer = Tokenizer(models.Unigram([("x", 0.0)]))
    tokenizer.normalizer = normalizers.Lowercase()
    before = tokenizer.encode("X").ids
    with pytest.raises(TypeError):
        tokenizer.normalizer = decoders.Unmark()
    with pytest.raises(ValueError):
        tokenizer.model = models.Unigram([("x", 0.0)], unk_id=9)
    assert tokenizer.encode("X").ids == before
    assert isinstance(tokenizer.normalizer, normalizers.Lowercase)
    tokenizer.normalizer = None
    assert tokenizer.normalizer is None
    tokenizer.pre_tokenizer = None
    tokenizer.decoder = None
    assert tokenizer.pre_tokenizer is None
    assert tokenizer.decoder is None
    with pytest.raises(TypeError):
        tokenizer.model = None
    path = tmp_path / "unchanged.json"
    tokenizer.save(path)
    assert json.loads(path.read_text())["model"]["vocab"] == [["x", 0.0]]


def test_fixed_digit_mismatches_and_unsafe_split_order_fail():
    trainer = trainers.MinGramTrainer(
        vocab_size=1600, fixed_digits="single", alphabet_mode="utf8", show_progress=False
    )
    tokenizer = create_claude_tokenizer(digits="triples")
    with pytest.raises(ValueError, match="digit|Digit"):
        tokenizer.train_from_iterator(["hello 1234567"], trainer=trainer)
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.FixedDigits.triples(),
        ]
    )
    with pytest.raises(ValueError, match="digit|Digit|split"):
        tokenizer.train_from_iterator(["hello 1234567"], trainer=trainers.MinGramTrainer(vocab_size=1600))


def test_explicit_configuration_is_not_overwritten_by_training():
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = decoders.Fuse()
    tokenizer.train_from_iterator(
        ["HELLO WORLD"] * 3, trainer=trainers.UnigramTrainer(vocab_size=50, show_progress=False)
    )
    assert isinstance(tokenizer.normalizer, normalizers.Lowercase)
    assert isinstance(tokenizer.pre_tokenizer, pre_tokenizers.WhitespaceSplit)
    assert isinstance(tokenizer.decoder, decoders.Fuse)
    assert tokenizer.decode(tokenizer.encode("HELLO").ids) == "hello"


def test_common_normalizers_and_component_snapshot_semantics():
    tokenizer = Tokenizer(models.Unigram([("é", 0.0), ("a", 0.0)]))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.decoder = decoders.Fuse()
    assert tokenizer.decode(tokenizer.encode("e\u0301").ids) == "é"
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
        ]
    )
    assert tokenizer.decode(tokenizer.encode("Á").ids) == "a"
    snapshot = tokenizer.normalizer
    with pytest.raises(AttributeError):
        snapshot.normalizers = []
    assert tokenizer.decode(tokenizer.encode("Á").ids) == "a"
    model = models.Unigram([("é", -1.5)])
    assert pickle.loads(pickle.dumps(model)) == model


def test_bpe_cannot_learn_over_a_fixed_inventory():
    tokenizer = create_claude_tokenizer(digits="triples")
    with pytest.raises(ValueError, match="Unigram|MinGram|BPE|fixed"):
        tokenizer.train_from_iterator(["hello 1234567"], trainer=trainers.BpeTrainer(vocab_size=1600))


def test_changing_preprocessing_preserves_trained_model_scores(tmp_path):
    tokenizer = Tokenizer(
        models.Unigram(
            [
                ("a", -3.8403830528259277),
                ("b", -13.5321998596191),
            ]
        )
    )
    before = tmp_path / "before.json"
    tokenizer.save(before)
    original = json.loads(before.read_text())["model"]
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = decoders.Fuse()
    after = tmp_path / "after.json"
    tokenizer.save(after)
    assert json.loads(after.read_text())["model"] == original
