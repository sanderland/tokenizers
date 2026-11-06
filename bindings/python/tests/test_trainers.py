from tokenizers import Tokenizer, trainers


def test_mingram_iterator_and_save(tmp_path):
    trainer = trainers.MinGramTrainer(vocab_size=400, show_progress=False)
    tokenizer = trainer.train_from_iterator(iter(["hello world", "hello world again"]))
    for text in ["hello world", "unseen 🦊"]:
        assert tokenizer.decode(tokenizer.encode(text).ids) == text
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = Tokenizer.from_file(path)
    assert loaded.decode(loaded.encode("unseen 🦊").ids) == "unseen 🦊"


def test_bpe_training_files(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("hello world\nhello world\n", encoding="utf-8")
    tokenizer = trainers.BpeTrainer(vocab_size=100, show_progress=False).train([path])
    assert tokenizer.decode(tokenizer.encode("hello world").ids) == "hello world"


def test_unigram_legacy_seed():
    tokenizer = trainers.UnigramTrainer(
        vocab_size=400, show_progress=False, seed_init="substrings"
    ).train_from_iterator(["hello world", "hello world"])
    assert tokenizer.decode(tokenizer.encode("hello world").ids) == "hello world"


def test_tokenizer_in_place_training():
    tokenizer = Tokenizer()
    result = tokenizer.train_from_iterator(
        ["hello world", "hello world"],
        trainer=trainers.MinGramTrainer(vocab_size=400, show_progress=False),
    )
    assert result is None
    assert tokenizer.decode(tokenizer.encode("unseen 🦊").ids) == "unseen 🦊"


def test_byte_level_unigram_and_mingram():
    for cls in (trainers.UnigramTrainer, trainers.MinGramTrainer):
        trainer = cls(vocab_size=300, alphabet_mode="byte_level", show_progress=False)
        tokenizer = trainer.train_from_iterator(["hello world", "hello there"])
        for text in ("hello world", "unseen 🦊", "\x00\n"):
            assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_retraining_preserves_preprocessing(tmp_path):
    import json

    trainer = trainers.BpeTrainer(vocab_size=100, show_progress=False)
    original = trainer.train_from_iterator(["hello world", "hello"])
    path = tmp_path / "source.json"
    original.save(path)
    config = json.loads(path.read_text())
    config["normalizer"] = {"type": "Lowercase"}
    path.write_text(json.dumps(config))
    tokenizer = Tokenizer.from_file(path)
    tokenizer.train_from_iterator(["HELLO THERE", "HELLO THERE"], trainer=trainer)
    assert tokenizer.decode(tokenizer.encode("HELLO THERE").ids) == "hello there"


def test_fixed_digit_inventories(tmp_path):
    import json

    for cls in (trainers.UnigramTrainer, trainers.MinGramTrainer):
        for mode, count in (("single", 10), ("single_marked", 40), ("triples", 1110)):
            tokenizer = cls(
                vocab_size=1600,
                show_progress=False,
                fixed_digits=mode,
                alphabet_mode="legacy",
            ).train_from_iterator(["hello 1234567 world", "hello 1234567 world"])
            path = tmp_path / "digits.json"
            tokenizer.save(path)
            config = json.loads(path.read_text())
            vocab = [entry[0] for entry in config["model"]["vocab"]]
            fixed = [v for v in vocab if v.strip("\ufdd0\ufdd1").isascii() and v.strip("\ufdd0\ufdd1").isdigit()]
            assert len(fixed) == count
            assert tokenizer.decode(tokenizer.encode("hello 1234567 world").ids) == "hello 1234567 world"
            assert "1234567" not in vocab


def test_byte_level_digits_survive_serialization(tmp_path):
    tokenizer = trainers.MinGramTrainer(
        vocab_size=1400, alphabet_mode="byte_level", fixed_digits="triples", show_progress=False
    ).train_from_iterator(["number 1234567", "number 1234567"])
    path = tmp_path / "byte-digits.json"
    tokenizer.save(path)
    loaded = Tokenizer.from_file(path)
    text = "number 001234567 🦊"
    assert loaded.decode(loaded.encode(text).ids) == text


def test_byte_level_bpe_round_trips_unseen_unicode(tmp_path):
    trainer = trainers.BpeTrainer(vocab_size=300, alphabet_mode="byte_level", show_progress=False)
    tokenizer = trainer.train_from_iterator(["hello world", "hello there"])
    path = tmp_path / "bpe.json"
    tokenizer.save(path)
    loaded = Tokenizer.from_file(path)
    for text in ("hello there", "unseen 🦊", "\x00\n"):
        assert loaded.decode(loaded.encode(text).ids) == text


def test_failed_retraining_keeps_pipeline_and_metadata(tmp_path):
    import json

    import pytest

    trainer = trainers.BpeTrainer(vocab_size=100, show_progress=False)
    source = trainer.train_from_iterator(["x x", "x x"])
    path = tmp_path / "with-postprocessor.json"
    source.save(path)
    config = json.loads(path.read_text())
    x_id = config["model"]["vocab"]["x"]
    config["post_processor"] = {
        "type": "TemplateProcessing",
        "single": [{"ids": [x_id]}, {"seq": "A"}],
        "pair": [{"ids": [x_id]}, {"seq": "A"}, {"seq": "B", "type_id": 1}],
    }
    path.write_text(json.dumps(config))
    tokenizer = Tokenizer.from_file(path)
    before = tokenizer.encode("x").ids
    with pytest.raises(ValueError, match="missing post-processor token"):
        tokenizer.train_from_iterator(["yyyyy", "yyyyy"], trainer=trainer)
    assert tokenizer.encode("x").ids == before
    after = tmp_path / "after.json"
    tokenizer.save(after)
    assert json.loads(after.read_text())["post_processor"] == config["post_processor"]


def test_bpe_rejects_unsupported_fixed_digits():
    import pytest

    trainer = trainers.BpeTrainer(vocab_size=300, fixed_digits="single", show_progress=False)
    with pytest.raises(ValueError, match="requires a Unigram or MinGram trainer"):
        trainer.train_from_iterator(["hello 123"])


def test_utf8_byte_alphabet_has_exactly_243_entries(tmp_path):
    import json

    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {b: chr(b) for b in printable}
    for index, byte in enumerate(b for b in range(256) if b not in mapping):
        mapping[byte] = chr(256 + index)
    expected = {mapping[b] for b in range(256) if b <= 0xBF or 0xC2 <= b <= 0xF4}
    assert len(expected) == 243
    for cls in (trainers.BpeTrainer, trainers.UnigramTrainer, trainers.MinGramTrainer):
        tokenizer = cls(vocab_size=243, alphabet_mode="byte_level", show_progress=False).train_from_iterator(["hello"])
        path = tmp_path / "243.json"
        tokenizer.save(path)
        vocab = json.loads(path.read_text())["model"]["vocab"]
        spellings = set(vocab) if isinstance(vocab, dict) else {entry[0] for entry in vocab}
        assert spellings == expected
        loaded = Tokenizer.from_file(path)
        for text in ("unseen 🦊", "\x00\n", "\u0080\u07ff\u0800\uffff\U00010000\U0010ffff"):
            assert loaded.decode(loaded.encode(text).ids) == text


def test_impossible_utf8_bytes_cannot_expand_initial_alphabet():
    import pytest

    for cls in (trainers.BpeTrainer, trainers.UnigramTrainer, trainers.MinGramTrainer):
        for symbol in ("À", "Á", "õ", "ÿ"):
            with pytest.raises(ValueError, match="mapped UTF-8 bytes"):
                cls(
                    vocab_size=300, alphabet_mode="byte_level", initial_alphabet=[symbol], show_progress=False
                ).train_from_iterator(["hello"])
