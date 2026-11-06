//! JSON-lines probe used by scripts/review_mingram.py. No network or corpus downloads.
use serde_json::{Value, json};
use std::io::{self, BufRead};
use tk_encode::decoders::{Decoder, unmark::Unmark};
use tk_encode::models::unigram::{Objective, Unigram};
use tk_encode::normalizers::{ClaudeNormalizer, MarkBoundaries};
use tk_encode::pre_tokenizers::mark_split::MarkSplit;
use tk_train::trainers::{AlphabetMode, MinGramTrainer, PruneCriterion, SeedInit};

fn probe(v: &Value) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    match v["op"].as_str().unwrap_or("mark") {
        "mark" => {
            let mode = v["mode"].as_str().unwrap_or("v47");
            let marking = match mode {
                "v3" => MarkBoundaries::v3(),
                "fixed" => MarkBoundaries::default(),
                _ => MarkBoundaries::v47(),
            };
            let cleaned = ClaudeNormalizer::new(mode != "v47").clean(v["text"].as_str().unwrap());
            let marked = marking.marked_stream(&cleaned);
            let spans: Vec<_> = MarkSplit
                .split(&marked)
                .iter()
                .map(|&(s, e)| marked[s as usize..e as usize].to_owned())
                .collect();
            let decoded = Unmark::new().decode(spans.clone())?;
            Ok(json!({"cleaned": cleaned, "marked": marked, "spans": spans, "decoded": decoded}))
        }
        "fixed" => {
            let pieces: Vec<(String, f64)> = serde_json::from_value(v["pieces"].clone())?;
            let model = Unigram::from(pieces, None, true)?.with_objective(Objective::MinTokens);
            let texts: Vec<String> = serde_json::from_value(v["texts"].clone())?;
            let encoded: Vec<_> = texts
                .iter()
                .map(|s| {
                    model
                        .tokenize(s)
                        .map(|tokens| tokens.into_iter().map(|t| t.value).collect::<Vec<_>>())
                })
                .collect::<Result<_, _>>()?;
            Ok(json!({"encoded": encoded}))
        }
        "train" | "em" | "trim" => {
            let mut builder = MinGramTrainer::builder();
            builder.show_progress(false);
            if v["alphabet_mode"] == "byte_level" {
                builder.alphabet_mode(AlphabetMode::ByteLevel);
            }
            if v["alphabet_mode"] == "legacy" {
                builder.alphabet_mode(AlphabetMode::Legacy);
            }
            if let Some(n) = v["vocab_size"].as_u64() {
                builder.vocab_size(n as u32);
            }
            if let Some(n) = v["seed_size"].as_u64() {
                builder.seed_size(n as usize);
            }
            if let Some(n) = v["seed_factor"].as_f64() {
                builder.seed_factor(n);
            }
            if let Some(n) = v["iterations"].as_u64() {
                builder.n_sub_iterations(n as u32);
            }
            if let Some(n) = v["max_length"].as_u64() {
                builder.max_piece_length(n as usize);
            }
            if let Some(n) = v["shrink"].as_f64() {
                builder.pruning_shrinking_factor(n);
            }
            if let Some(b) = v["skip_substring"].as_bool() {
                builder.skip_substring_in_batch(b);
            }
            if v["seed"] == "substrings" {
                builder.seed_init(SeedInit::Substrings);
            }
            if v["criterion"] == "mi" {
                builder.prune_criterion(PruneCriterion::MinimumIncrease);
            }
            let trainer = builder.build()?;
            let sentences: Vec<(String, u32)> = serde_json::from_value(v["sentences"].clone())?;
            if v["op"] == "em" {
                let pieces: Vec<String> = serde_json::from_value(v["pieces"].clone())?;
                return Ok(json!({"pieces": trainer.em_scores(&pieces, &sentences)?}));
            }
            if v["op"] == "trim" {
                let pieces: Vec<(String, f64)> = serde_json::from_value(v["pieces"].clone())?;
                return Ok(
                    json!({"pieces": trainer.mingram_pp_trim(&pieces, &sentences, v["target"].as_u64().unwrap() as usize)?}),
                );
            }
            let mut model = Unigram::default();
            trainer.do_train(sentences, &mut model)?;
            Ok(json!({"pieces": model.vocab(), "size": model.len()}))
        }
        op => Err(format!("unknown op {op}").into()),
    }
}

fn main() {
    // One response per request, including errors, so Python can report all failures.
    for line in io::stdin().lock().lines() {
        let result = line
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
            .and_then(|v| probe(&v).map_err(|e| e.to_string()));
        println!(
            "{}",
            match result {
                Ok(v) => v,
                Err(e) => json!({"error": e}),
            }
        );
    }
}
