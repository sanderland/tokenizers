#!/usr/bin/env python3
"""UTF-8 compression checks using UDHR or Rosetta fixtures in sibling ctok.

This is a CPU sanity experiment, not a reproduction of the paper's FineWeb/
Goldfish results. Every fifth nonempty line is held out. No corpus downloads.
Run after building the review_mingram example, with script_tok's dependencies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import time

from review_mingram import ROOT, rust
from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import UTF8PretokenizerConfig, GPT4O_REGEX
from script_bpe.pretokenize.pretokenizer import UTF8Pretokenizer
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from script_bpe.tokenizers.mingram.trainer import MinGramTrainer, MinGramTrainerConfig
from script_bpe.tokenizers.mingram.model import MinGramModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--corpus", choices=["udhr", "rosetta"], default="udhr")
    parser.add_argument("--factors", type=float, nargs="+", default=[1.1, 2.0])
    args = parser.parse_args()
    source = ROOT.parent / f"ctok/tests/fixtures/{args.corpus}.jsonl.gz"
    wanted = {"eng", "deu_1996", "fin", "rus", "arb", "kor"}
    with gzip.open(source, "rt") as f:
        records = [json.loads(line) for line in f]
    holdout_source = None
    if args.corpus == "udhr":
        docs = [r for r in records if r["f"] in wanted]
        assert {r["f"] for r in docs} == wanted
    else:
        holdout_source = source.with_name("rosetta_holdout.jsonl.gz")
        with gzip.open(holdout_source, "rt") as f:
            holdout = [json.loads(line) for line in f]
        assert not ({r["id"] for r in records} & {r["id"] for r in holdout})
        docs = [{"f": "rosetta", "train": [r["text"] for r in records], "test": [r["text"] for r in holdout]}]
    pt = UTF8Pretokenizer(UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX, remove_unassigned=False))
    results = []
    with tempfile.TemporaryDirectory(prefix="mingram-compression-") as tmp, args.output.with_suffix(".log").open("w") as log:
        for doc in docs:
            if "train" in doc:
                train, test = doc["train"], doc["test"]
            else:
                lines = [s for s in doc["text"].splitlines() if s.strip()]
                train = [s for i, s in enumerate(lines) if i % 5 != 0]
                test = [s for i, s in enumerate(lines) if i % 5 == 0]
            sentences = Counter(pt.decode(chunk, errors="strict") for text in train for chunk in pt.pretokenize(text))
            test_chunks = [pt.decode(chunk, errors="strict") for text in test for chunk in pt.pretokenize(text)]
            char_set = set("".join(sentences)) | {chr(i) for i in range(128)} | {chr(i) for i in range(0xFDD0, 0xFDD5)}
            prefixes = {c.encode()[:n] for c in char_set for n in range(2, len(c.encode()))}
            reserved = 115 + len(prefixes)
            learned = args.vocab_size - reserved
            assert learned >= len(char_set), (doc["f"], learned, len(char_set))
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                corpus = PretokenizedCorpus.from_texts("udhr_" + doc["f"], train, pt, base_path=tmp)
                start = time.perf_counter()
                bpe = BPETrainer(pt, corpus, BPETrainerConfig(additional_vocab_size=args.vocab_size - 256,
                                                           num_workers=1, verbose=False)).train()
                baseline = sum(len(bpe.encode(text)) for text in test)
                bpe_time = time.perf_counter() - start
            row = {"language": doc["f"], "train_sequences": len(train), "test_sequences": len(test),
                   "train_utf8_bytes": sum(len(pt.normalize(s).encode()) for s in train),
                   "test_utf8_bytes": sum(len(pt.normalize(s).encode()) for s in test),
                   "reference_bpe": {"size": len(bpe.tokens), "test_tokens": baseline, "seconds": bpe_time},
                   "rust_required_characters": len(char_set), "rust_fallback_pieces": reserved,
                   "variants": []}
            for factor in args.factors:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    start = time.perf_counter()
                    model = MinGramTrainer(pt, corpus, MinGramTrainerConfig(
                        additional_vocab_size=args.vocab_size - 256, overshoot_factor=factor,
                        num_workers=1, verbose=False)).train()
                    raw_reference_count = sum(len(model.encode(s)) for s in test)
                    # The research evaluation loads saved models. Its loader
                    # floors unused atomic -inf scores to -100. Directly
                    # encoding the freshly trained object can give different
                    # held-out paths, so use that production load path here.
                    path = str(Path(tmp) / f"{doc['f']}_{factor}.model.json.gz")
                    model.save(path)
                    model = MinGramModel.load(path)
                    reference_count = sum(len(model.encode(s)) for s in test)
                    reference_time = time.perf_counter() - start
                start = time.perf_counter()
                rs = rust([{"op": "train", "sentences": sorted(sentences.items()),
                            "vocab_size": args.vocab_size,
                            "seed_factor": factor}])[0]
                if "error" in rs:
                    raise RuntimeError(rs)
                enc = rust([{"op": "fixed", "pieces": rs["pieces"], "texts": test_chunks}])[0]
                if "error" in enc:
                    raise RuntimeError(enc)
                rust_count = sum(map(len, enc["encoded"]))
                rust_time = time.perf_counter() - start
                row["variants"].append({"factor": factor,
                    "reference": {"size": len(model.tokens), "test_tokens": reference_count,
                                  "freshly_trained_test_tokens": raw_reference_count,
                                  "delta_vs_bpe_pct": 100 * (reference_count / baseline - 1), "seconds": reference_time},
                    "rust": {"size": rs["size"], "test_tokens": rust_count,
                             "delta_vs_reference_bpe_pct": 100 * (rust_count / baseline - 1), "seconds": rust_time}})
            results.append(row)
            print(json.dumps(row), flush=True)
    report = {"source": f"ctok/tests/fixtures/{source.name}",
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "holdout_source": str(holdout_source.relative_to(ROOT.parent)) if holdout_source else None,
              "holdout_sha256": hashlib.sha256(holdout_source.read_bytes()).hexdigest() if holdout_source else None,
              "split": ("1741 training documents and the separate 250-document holdout; verified disjoint IDs."
                        if args.corpus == "rosetta" else "Every fifth nonempty line, starting at index 0, held out; remaining lines train."),
              "vocab_size": args.vocab_size, "pretokenizer": "NFC + GPT4O regex + UTF-8 character boundaries",
              "caveats": ["These are local fixture corpora, not the paper's FineWeb/Goldfish corpora or results.",
                          "Rust seeds on characters; script_tok learns UTF-8 prefixes and characters from byte atoms.",
                          "Reserved alphabets differ, and Rust limits pieces to 16 characters; vocabulary caps match, actual returned sizes are recorded.",
                          "Reference MinGram is saved and loaded before evaluation, as its loader floors unused atomic scores.",
                          "Rust timings use a debug build, include subprocess overhead, and are not a speed benchmark."],
              "results": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
