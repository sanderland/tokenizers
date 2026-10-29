//! Train a Claude-style tokenizer, then improve on it.
//!
//! Training sees the same pre-token spans inference tiles: each corpus line
//! is cleaned, marked, and split with `MarkSplit`, and the trainer learns
//! marked pieces that never cross a boundary mark. Two configs share the
//! recipe and differ in one flag:
//!
//! - v4.7-faithful (`marks_part_of_words: false`): marks split words, as
//!   Claude does (the mark-breaking bug).
//! - fixed (`marks_part_of_words: true`): marks stay inside their words.
//!
//! Newer (v4.7+) boundaries only: `ClaudeNormalizer { fold_quotes: false }`
//! plus `MarkBoundaries::v47` plus `MarkSplit`, with the byte floor always
//! on. Framing (message pads, head-space absorption) is chat templating,
//! not shown here.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p tk-train --example claude_style --features normalizers
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use tk_encode::decoders::DecoderRuntime;
use tk_encode::decoders::unmark::Unmark;
use tk_encode::models::unigram::Unigram;
use tk_encode::normalizers::{ClaudeNormalizer, MarkBoundaries};
use tk_encode::pipeline::{
    EncodeOptions, PipelineModel, PipelineNormalizer, PipelinePostProcessor, PipelinePreTokenizer,
    PipelineToken, PipelineTokenizer,
};
use tk_encode::pre_tokenizers::mark_split::MarkSplit;
use tk_encode::vocab::bucket_added_vocabulary::AddedVocabulary;
use tk_train::trainers::{MinGramTrainer, SeedInit};

/// Tiny multilingual corpus: English plus Thai and Devanagari words whose
/// marks (tone marks, virama) survive NFC and trigger the mark-breaking bug
/// in faithful mode.
fn corpus() -> Vec<(String, u32)> {
    [
        ("the quick brown fox jumps over the lazy dog", 20),
        ("hello world", 15),
        ("tokenization is fun", 10),
        ("hello tokenization world", 8),
        ("the fox says hello", 8),
        // น้ำ (water), ข้าว (rice), แล้ว (already)
        ("น้ำ", 12),
        ("ข้าว", 12),
        ("แล้ว", 12),
        ("น้ำ ข้าว แล้ว", 8),
        ("hello น้ำ world", 6),
        // हिन्दी (Hindi), नमस्ते (namaste)
        ("हिन्दी", 12),
        ("नमस्ते", 12),
        ("हिन्दी नमस्ते", 8),
        ("hello हिन्दी world", 6),
    ]
    .into_iter()
    .map(|(text, count)| (text.to_string(), count))
    .collect()
}

/// Training sentences: cleaned, marked, and split into pre-token spans.
/// Seeding and EM see spans, not whole lines, so no learned piece ever
/// crosses a boundary mark; inference runs the same split before the
/// lattice. Each span inherits its line's count.
fn marked_sentences(marking: &MarkBoundaries) -> Vec<(String, u32)> {
    let cleanup = ClaudeNormalizer::new(false);
    let splitter = MarkSplit;
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for (line, count) in corpus() {
        let marked = marking.marked_stream(&cleanup.clean(&line));
        for (s, e) in splitter.split(&marked) {
            *counts
                .entry(marked[s as usize..e as usize].to_string())
                .or_default() += count;
        }
    }
    counts.into_iter().collect()
}

fn train(marking: &MarkBoundaries) -> Result<Unigram, Box<dyn Error + Send + Sync>> {
    let trainer = MinGramTrainer::builder()
        .show_progress(false)
        .vocab_size(400)
        .seed_size(500)
        .seed_init(SeedInit::Bpe)
        .build()
        .unwrap();
    let mut model = Unigram::default();
    trainer.do_train(marked_sentences(marking), &mut model)?;
    Ok(model)
}

fn pipeline_for(model: Unigram, marking: MarkBoundaries) -> PipelineTokenizer {
    PipelineTokenizer::from_parts(
        AddedVocabulary::new(),
        vec![
            PipelineNormalizer::ClaudeNormalizer(ClaudeNormalizer::new(false)),
            PipelineNormalizer::MarkBoundaries(marking),
        ],
        PipelinePreTokenizer::MarkSplit(MarkSplit),
        PipelineModel::Unigram(model),
        PipelinePostProcessor::default(),
        Some(DecoderRuntime::Unmark(Unmark::new())),
        BTreeMap::default(),
        None,
        None,
    )
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut fixed_marking = MarkBoundaries::v47();
    fixed_marking.marks_part_of_words = true;

    // 1. Train once per boundary rule; each vocab matches its inference.
    let faithful_model = train(&MarkBoundaries::v47())?;
    let fixed_model = train(&fixed_marking)?;
    println!(
        "trained vocabs: faithful {} pieces, fixed {} pieces",
        faithful_model.len(),
        fixed_model.len()
    );

    // 2. Save and reload both configs.
    let dir = tempfile::tempdir()?;
    for (name, model, marking) in [
        ("faithful", faithful_model, MarkBoundaries::v47()),
        ("fixed", fixed_model, fixed_marking),
    ] {
        let pipeline = pipeline_for(model, marking);
        let path = dir.path().join(format!("claude_v47_{name}.json"));
        std::fs::write(&path, tk_serialize::to_json(&pipeline)?)?;
        println!("wrote {}", path.display());
    }

    let faithful = tk_serialize::from_json_file(dir.path().join("claude_v47_faithful.json"))?;
    let fixed = tk_serialize::from_json_file(dir.path().join("claude_v47_fixed.json"))?;
    let options = EncodeOptions::no_specials();

    // 3. Compare on mark-heavy text. The fix wins where the bug broke words
    // at marks; in-vocab ties are both sides absorbing their own units, and
    // plain text ties.
    let demos = [
        "hello world",
        "น้ำ ข้าว แล้ว",
        "แล้ว ข้าว น้ำ",
        "ข้าว น้ำ ข้าว",
        "हिन्दी नमस्ते",
        "नमस्ते हिन्दी",
        "नमस्ते हिन्दी नमस्ते",
        "hello น้ำ world",
        // Held-out mark words: never seen in training.
        "ไก่",
        "ไก่ น้ำ",
        "प्रेम",
        "प्रेम हिन्दी",
    ];
    let mut strictly_fewer = 0;
    let cleanup = ClaudeNormalizer::new(false);
    for demo in demos {
        let buggy = faithful.encode(vec![demo], &options).wait()?;
        let mended = fixed.encode(vec![demo], &options).wait()?;
        let (b, m) = (buggy[0].len(), mended[0].len());
        println!("{demo:?}: faithful {b} tokens, fixed {m} tokens");
        let tokens = fixed.decode_tokens(PipelineToken::cast_slice(mended[0].ids()), false);
        println!("  fixed tokens: {tokens:?}");
        // Full round-trip through tiling, bytes, and the decoder.
        for (pipeline, encoding) in [(&faithful, &buggy[0]), (&fixed, &mended[0])] {
            let back = pipeline.decode(PipelineToken::cast_slice(encoding.ids()), false)?;
            assert_eq!(back, cleanup.clean(demo), "round-trip of {demo:?}");
        }
        println!("  decodes to: {:?}", cleanup.clean(demo));
        strictly_fewer += usize::from(m < b);
    }
    assert!(
        strictly_fewer > 0,
        "expected the fix to win on at least one demo"
    );
    println!("the fix wins on {strictly_fewer} of {} demos", demos.len());
    Ok(())
}
