#!/usr/bin/env python3
"""Reproducible local audit against sibling ctok and script_tok checkouts.

Build: cd tokenizers && cargo build -p tk-train --example review_mingram --features normalizers
Run: python scripts/review_mingram.py --output mingram-review-results.json
Requires script_tok's runtime dependencies and regex. No downloaded corpora.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parent / "ctok"), str(ROOT.parent / "script_tok")]

from ctok.normalize import nfc, stream_norm
from script_bpe.pretokenize import UTF8PretokenizerConfig
from script_bpe.pretokenize.pretokenizer import UTF8Pretokenizer
from script_bpe.tokenizers.mingram.model import MinGramModel
from script_bpe.tokenizers.mingram.trainer import MinGramTrainer, MinGramTrainerConfig
from script_bpe.tokenizers.unigram.model import UnigramToken
from script_bpe.utils import token_array


def rust(requests):
    p = subprocess.run([str(ROOT / "tokenizers/target/debug/examples/review_mingram")],
                       input="".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests),
                       text=True, capture_output=True, check=True)
    rows = [json.loads(line) for line in p.stdout.splitlines()]
    assert len(rows) == len(requests), p.stderr
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(20261106)
    examples = ["", "hello world", "Hello NASA HELLO WiFi", "1234567890", "12 34", "a123b",
                "123.456", "١٢٣ ٤٥٦", "१२३ ४५६", "１２３ ４５６", "²³ ⁴⁵", "123abc456",
                "it's don't I'M", "a!b", "a! b", "a  b", " a", "a\n\n", "\r\n",
                "ọ̀rọ̀", "राष्ट्रीय", "น้ำ ข้าว แล้ว", "हिन्दी नमस्ते", "日本語 中文 한글",
                "a✈️b", "𐐀𐐨 𓀀 😀", "e\u0301", "a\0b", "a\uE000b", "a\uFDD0b",
                "ǅabc", "ᾈα", "ϴϓℳ", "İstanbul", "ΟΣ", "a\u0D3Bb", "Აბ", "ᲐᲑᲒᲓ"]
    alphabet = "abABΣσςİẞ123١२３  \t\n.,'’ọ्̀ข้中한😀\uFDD0"
    texts = examples + ["".join(rng.choices(alphabet, k=rng.randrange(1, 22))) for _ in range(600)]
    # Deterministic BMP coverage, emphasizing marks and case mappings.
    interesting = [chr(i) for i in range(0x10000) if unicodedata.category(chr(i)) in {"Mn", "Mc", "Me", "Lt"}]
    interesting += [chr(i) for i in range(0x10000)
                    if i % 11 == 0 and unicodedata.category(chr(i))[0] in {"L", "N", "P", "S"}]
    texts += ["a" + c + "b" for c in interesting]
    req = [{"op": "mark", "mode": m, "text": t} for m in ("v3", "v47", "fixed") for t in texts]
    rows = rust(req)
    differences, roundtrips = [], []
    for r, got in zip(req, rows):
        text, mode = r["text"], r["mode"]
        cleaned = nfc(text, fold_quotes=mode != "v47")
        if got["decoded"] != got["cleaned"]:
            roundtrips.append({**r, **got})
        if mode != "fixed":
            ref = stream_norm(cleaned, SimpleNamespace(frame_bow=False, allcaps_min=4 if mode == "v3" else None))
            # ctok drops content-final newlines for its separate frame ladder.
            if got["cleaned"] != cleaned or got["marked"].rstrip("\n") != ref:
                differences.append({**r, **got, "reference": ref})

    # Actual script_tok byte-DP versus Rust on an identical fixed UTF-8 vocab.
    pt = UTF8Pretokenizer(UTF8PretokenizerConfig(normalization=None, remove_unassigned=False))
    strings = ["a", "b", "c", "日", "本", "語", "한", "글", "é", "😀"]
    strings += ["".join(rng.choices(strings[:7], k=rng.randrange(2, 5))) for _ in range(100)]
    strings = list(dict.fromkeys(strings))
    scores = [-rng.randrange(1, 25) / 3 for _ in strings]
    py_model = MinGramModel(pt, [UnigramToken(id=i, atomic_tokens=token_array(pt.byte_ids[b] for b in s.encode()),
                          log_prob=score, required=len(s.encode()) == 1)
                          for i, (s, score) in enumerate(zip(strings, scores))])
    trials = ["".join(rng.choices(strings[:10], k=rng.randrange(1, 20))) for _ in range(500)]
    actual = rust([{"op": "fixed", "pieces": list(zip(strings, scores)), "texts": trials}])[0]["encoded"]
    parity = [{"text": t, "python": len(py_model.encode(t)), "rust": len(out)}
              for t, out in zip(trials, actual) if len(py_model.encode(t)) != len(out)]

    # Fixed-vocab hard EM: ASCII isolates algorithm changes from byte fallback.
    words = [("ab", 3), ("a", 1), ("abc", 2), ("bca", 4), ("ababc", 2)]
    vocab = ["a", "b", "c", "ab", "bc", "abc"]
    corpus = [(pt.pretokenize(w)[0], freq) for w, freq in words]
    trainer = MinGramTrainer(pt, corpus, MinGramTrainerConfig(additional_vocab_size=3, verbose=False))
    model = MinGramModel(pt, [UnigramToken(id=i, atomic_tokens=token_array(w.encode()[j] + 1 for j in range(len(w))),
                           log_prob=0.0, required=len(w) == 1) for i, w in enumerate(vocab)])
    em = []
    for iteration in range(2):
        counts, _, _ = trainer.run_e_step(model, sum(len(w) * f for w, f in words))
        em.append(dict(counts))
        model, _ = trainer.run_m_step(model, counts)
    got_scores = rust([{"op": "em", "pieces": vocab, "sentences": words, "iterations": 2}])[0]
    rust_model = MinGramModel(pt, [UnigramToken(id=i, atomic_tokens=token_array(b + 1 for b in w.encode()),
                             log_prob=score, required=len(w) == 1) for i, (w, score) in enumerate(got_scores["pieces"])])
    _, _, py_count = trainer.run_e_step(model, sum(len(w) * f for w, f in words))
    _, _, rs_count = trainer.run_e_step(rust_model, sum(len(w) * f for w, f in words))

    # Direct regression probes for the advertised vocabulary limit and unsafe options.
    configs = [{"label": "exact_floor_budget", "vocab_size": 249},
               {"label": "one_spare_slot", "vocab_size": 250},
               {"label": "zero_iterations", "iterations": 0},
               {"label": "zero_max_length", "max_length": 0},
               {"label": "invalid_shrink", "shrink": 2.0}]
    probes = [{"op": "train", "sentences": [("abc", 20), ("ab", 10)], "seed_size": 300,
               "vocab_size": 300, **c} for c in configs]
    trained = rust(probes)
    budgets = [{"config": c, **{k: v for k, v in got.items() if k != "pieces"}}
               for c, got in zip(configs, trained)]
    # Prefix guarantee: a trained prefix can help an unseen character sharing it.
    prefix_req = {"op": "train", "sentences": [("日本語", 20), ("한글", 10), ("😀", 10)],
                  "vocab_size": 400, "seed_size": 500}
    prefix_model = rust([prefix_req])[0]
    unknown = rust([{"op": "fixed", "pieces": prefix_model["pieces"], "texts": ["😀", "😁", "𓀀", "日", "한"]}])[0]
    report = {
        "seed": 20261106, "python_unicode": unicodedata.unidata_version,
        "normalization_cases": len(req), "ctok_comparisons": 2 * len(texts),
        "ctok_differences": differences, "roundtrip_failures": roundtrips,
        "digit_examples": [{"text": r["text"], **got} for r, got in zip(req, rows)
                           if r["mode"] == "v47" and r["text"] in examples[3:13]],
        "fixed_utf8_comparisons": len(trials), "fixed_utf8_count_differences": parity,
        "hard_em": {"python_counts": em, "python_final_corpus_tokens": py_count,
                    "rust_final_corpus_tokens": rs_count, "rust_scores": got_scores},
        "configuration_probes": budgets, "fallback_examples": unknown,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in report.items()
                      if k not in {"hard_em", "fallback_examples"}}, indent=2))


if __name__ == "__main__":
    main()
