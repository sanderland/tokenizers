#!/usr/bin/env python3
"""Reproduce the local UTF-8 MinGram comparison through the Python trainer API.

Uses the sibling ctok Rosetta fixtures and script_tok pretokenizer. The pinned
reference report records a separate run of script_tok with the same 243 atoms.
This is not the Claude recipe or a reproduction of paper-scale experiments.
"""

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

from compare_mingram_compression import (
    GPT4O_REGEX,
    ROOT,
    UTF8Pretokenizer,
    UTF8PretokenizerConfig,
)

from tokenizers import Tokenizer, decoders, models, normalizers, trainers


def texts(path):
    with gzip.open(path, "rt") as source:
        return [json.loads(line)["text"] for line in source]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/experiments/ctok-mingram-python.json",
    )
    args = parser.parse_args()
    reference = json.loads(
        (ROOT / "docs/experiments/ctok-mingram-byte-alphabet-243.json").read_text()
    )
    source = ROOT.parent / "ctok/tests/fixtures/rosetta.jsonl.gz"
    holdout = source.with_name("rosetta_holdout.jsonl.gz")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == reference["source_sha256"]
    assert (
        hashlib.sha256(holdout.read_bytes()).hexdigest() == reference["holdout_sha256"]
    )
    pt = UTF8Pretokenizer(
        UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX, remove_unassigned=False)
    )
    train_docs, test_docs = texts(source), texts(holdout)

    def chunks(docs):
        return (
            pt.decode(chunk, errors="strict")
            for doc in docs
            for chunk in pt.pretokenize(doc)
        )

    trainer = trainers.MinGramTrainer(
        vocab_size=4096,
        alphabet_mode="byte_level",
        max_piece_length=256,
        seed_factor=1.1,
        show_progress=False,
    )
    tokenizer = Tokenizer(models.Unigram([]))
    tokenizer.normalizer = normalizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(chunks(train_docs), trainer=trainer)
    test_chunks = list(chunks(test_docs))
    with tempfile.TemporaryDirectory(prefix="python-mingram-") as tmp:
        path = Path(tmp) / "tokenizer.json"
        tokenizer.save(path)
        config = json.loads(path.read_text())
        loaded = Tokenizer.from_file(path).compile()
        encoded = loaded.encode_batch(test_chunks)
        assert all(
            loaded.decode(enc.ids) == text
            for text, enc in zip(test_chunks, encoded, strict=True)
        )
    count = sum(map(len, encoded))
    size = len(config["model"]["vocab"])
    atomic_count = sum(len(piece) == 1 for piece, _ in config["model"]["vocab"])
    expected = reference["reference_243_mingram"]["test_tokens"]
    baseline = reference["reference_243_bpe"]["test_tokens"]
    assert size == 4096, size
    assert atomic_count == 243, atomic_count
    assert count == expected, (count, expected)
    result = {
        "source_sha256": reference["source_sha256"],
        "holdout_sha256": reference["holdout_sha256"],
        "train_documents": len(train_docs),
        "holdout_documents": len(test_docs),
        "pretokenizer": "NFC + GPT4O regex",
        "alphabet_size": atomic_count,
        "vocab_size": size,
        "max_piece_length_bytes": 256,
        "seed_factor": 1.1,
        "python_test_tokens": count,
        "script_tok_test_tokens": expected,
        "reference_bpe_test_tokens": baseline,
        "gain_vs_bpe_pct": 100 * (1 - count / baseline),
        "all_holdout_chunks_round_trip": True,
        "caveats": [
            "Reference counts come from the separately executed, hash-matched 243-byte comparison.",
            "This uses GPT4O regex chunks, not Claude preprocessing.",
            "Local fixture comparison, not a reproduction of the paper's full evaluation.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
