#!/usr/bin/env python3
"""Compare byte-alphabet MinGram with the existing Rosetta reference results."""

from collections import Counter
import contextlib, gzip, hashlib, json, tempfile
from pathlib import Path
from compare_mingram_compression import (
    ROOT,
    rust,
    UTF8Pretokenizer,
    UTF8PretokenizerConfig,
    GPT4O_REGEX,
    PretokenizedCorpus,
    BPETrainer,
    BPETrainerConfig,
    MinGramTrainer,
    MinGramTrainerConfig,
    MinGramModel,
)


def main():
    source = ROOT.parent / "ctok/tests/fixtures/rosetta.jsonl.gz"
    holdout = source.with_name("rosetta_holdout.jsonl.gz")

    def texts(path):
        with gzip.open(path, "rt") as f:
            return [json.loads(line)["text"] for line in f]

    train_texts, test_texts = texts(source), texts(holdout)
    pt = UTF8Pretokenizer(
        UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX, remove_unassigned=False)
    )
    # Same reversible byte alphabet as tk_encode::utils::byte_level.
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {b: chr(b) for b in printable}
    for index, b in enumerate(b for b in range(256) if b not in mapping):
        mapping[b] = chr(256 + index)

    def mapped(raw):
        return "".join(mapping[b] for b in raw)

    train = Counter(
        mapped(pt.token_to_bytes[a][0] for a in c)
        for s in train_texts
        for c in pt.pretokenize(s)
    )
    test = [
        mapped(pt.token_to_bytes[a][0] for a in c)
        for s in test_texts
        for c in pt.pretokenize(s)
    ]
    rows = []
    for length in [16, 64, 256]:
        model = rust(
            [
                {
                    "op": "train",
                    "alphabet_mode": "byte_level",
                    "vocab_size": 4096,
                    "seed_factor": 1.1,
                    "max_length": length,
                    "sentences": sorted(train.items()),
                }
            ]
        )[0]
        if "error" in model:
            raise RuntimeError(model)
        encoded = rust([{"op": "fixed", "pieces": model["pieces"], "texts": test}])[0]
        if "error" in encoded:
            raise RuntimeError(encoded)
        count = sum(map(len, encoded["encoded"]))

        def safe(s):
            raw = bytes({c: b for b, c in mapping.items()}[c] for c in s)
            try:
                raw.decode()
                return True
            except UnicodeDecodeError as e:
                return e.start == 0 and e.reason == "unexpected end of data"

        invalid = [s for s, _ in model["pieces"] if len(s) > 1 and not safe(s)]
        assert not invalid, invalid[:5]
        symbols = {p for p, _ in model["pieces"] if len(p) == 1}
        expected = {mapping[b] for b in range(256) if b <= 0xBF or 0xC2 <= b <= 0xF4}
        assert symbols == expected, (
            len(symbols),
            symbols - expected,
            expected - symbols,
        )
        row = {
            "max_piece_length_bytes": length,
            "actual_size": model["size"],
            "test_tokens": count,
            "gain_vs_reference_256_bpe_pct": 100 * (1 - count / 79722),
            "invalid_multibyte_pieces": len(invalid),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Keep byte IDs stable so normal UTF8Pretokenizer save/load remains valid;
    # only remove impossible atoms from the trainer's required inventory.
    excluded = {
        pt.byte_ids[b] for b in range(256) if not (b <= 0xBF or 0xC2 <= b <= 0xF4)
    }
    pt.atomic_tokens = {i: s for i, s in pt.atomic_tokens.items() if i not in excluded}
    assert len(pt.atomic_tokens) == 243
    with tempfile.TemporaryDirectory(prefix="mingram-243-") as tmp:
        with (
            (Path(tmp) / "training.log").open("w") as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            corpus = PretokenizedCorpus.from_texts(
                "rosetta_243", train_texts, pt, base_path=tmp
            )
            bpe = BPETrainer(
                pt,
                corpus,
                BPETrainerConfig(
                    additional_vocab_size=4096 - 243, num_workers=1, verbose=False
                ),
            ).train()
            reference_bpe = {
                "actual_size": len(bpe.tokens),
                "test_tokens": sum(len(bpe.encode(s)) for s in test_texts),
            }
            model = MinGramTrainer(
                pt,
                corpus,
                MinGramTrainerConfig(
                    additional_vocab_size=4096 - 243,
                    overshoot_factor=1.1,
                    num_workers=1,
                    verbose=False,
                ),
            ).train()
            path = str(Path(tmp) / "mingram.json.gz")
            model.save(path)
            model = MinGramModel.load(path)
            reference_mingram = {
                "actual_size": len(model.tokens),
                "test_tokens": sum(len(model.encode(s)) for s in test_texts),
            }
    for row in rows:
        row["gain_vs_reference_243_bpe_pct"] = 100 * (
            1 - row["test_tokens"] / reference_bpe["test_tokens"]
        )
    print(
        json.dumps(
            {
                "reference_243_bpe": reference_bpe,
                "reference_243_mingram": reference_mingram,
            }
        ),
        flush=True,
    )
    output = ROOT / "docs/experiments/ctok-mingram-byte-alphabet-243.json"
    output.write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "holdout_sha256": hashlib.sha256(holdout.read_bytes()).hexdigest(),
                "reference_256_bpe_test_tokens": 79722,
                "reference_256_mingram_test_tokens": 79045,
                "old_rust_utf8_test_tokens": 79542,
                "seed_factor": 1.1,
                "vocab_size": 4096,
                "byte_alphabet_size": 243,
                "reference_243_bpe": reference_bpe,
                "reference_243_mingram": reference_mingram,
                "results": rows,
                "caveats": [
                    "Same NFC and GPT4O regex chunks; Rust sees a reversible mapping of their UTF-8 bytes.",
                    "Rust BPE and script_tok BPE can choose different merges on equal frequencies.",
                    "Length counts byte atoms in this alphabet, not Unicode codepoints.",
                    "The original reference counts used 256 atoms; reference_243 removes the 13 impossible bytes from its required inventory while preserving byte IDs and using the standard save/load path.",
                    "This fixture is not the paper corpus.",
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
