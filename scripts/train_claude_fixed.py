#!/usr/bin/env python3
"""Train the fork's Claude-inspired recipe from UTF-8 text files."""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, trainers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument(
        "--algorithm", choices=["mingram", "unigram"], default="mingram"
    )
    parser.add_argument(
        "--digits",
        choices=["single", "single_marked", "triples", "learned"],
        default="single",
    )
    parser.add_argument("--max-piece-length", type=int, default=16)
    args = parser.parse_args()
    cls = (
        trainers.MinGramTrainer
        if args.algorithm == "mingram"
        else trainers.UnigramTrainer
    )
    trainer = cls(
        recipe="claude_fixed",
        alphabet_mode="utf8",
        fixed_digits=None if args.digits == "learned" else args.digits,
        vocab_size=args.vocab_size,
        max_piece_length=args.max_piece_length,
    )
    tokenizer = Tokenizer()
    tokenizer.train(args.files, trainer=trainer)
    tokenizer.save(args.output)
    loaded = Tokenizer.from_file(args.output)
    for sample in ["Hello 1234567", "नमस्ते हिन्दी", "น้ำ ข้าว"]:
        encoded = loaded.encode(sample)
        print(
            f"{sample!r}: {len(encoded)} tokens; decoded={loaded.decode(encoded.ids)!r}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
