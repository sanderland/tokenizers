//! Shared seeding and finalization for Unigram-family trainers.
//!
//! Both `UnigramTrainer` and `MinGramTrainer` start from a seed vocabulary and
//! finish by keeping required characters and laying out special tokens the same
//! way. The seed can be built from frequent substrings (the classic
//! suffix-array seed) or from a BPE training run.

use crate::progress::ProgressBar;
#[cfg(feature = "bpe")]
use crate::trainers::bpe::BpeTrainer;
use ahash::{AHashMap, AHashSet};
#[cfg(feature = "bpe")]
use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use tk_encode::Result;
#[cfg(feature = "bpe")]
use tk_encode::models::bpe::{BpeConfig, PipelineBPE};
use tk_encode::models::unigram::Unigram;
#[cfg(feature = "bpe")]
use tk_encode::pipeline::Model as _;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

// A token and a score
pub(crate) type SentencePiece = (String, f64);

// A full sentence or word + its count within the dataset
pub(crate) type Sentence = (String, u32);

/// Seed vocabulary cap when the trainer's `seed_size` is unset (upstream default).
pub(crate) const DEFAULT_SEED_SIZE: usize = 1_000_000;

/// How a Unigram-family trainer builds its seed vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedInit {
    /// Frequent substrings from a suffix array (classic seed).
    #[default]
    Substrings,
    /// Tokens of a BPE model trained to `seed_size`.
    Bpe,
}

pub(crate) fn to_log_prob(pieces: &mut [SentencePiece]) {
    let sum: f64 = pieces.iter().map(|(_, score)| score).sum();
    let logsum = sum.ln();
    for (_, score) in pieces.iter_mut() {
        *score = score.ln() - logsum;
    }
}

fn is_valid_sentencepiece(char_string: &[char], max_piece_length: usize) -> bool {
    // Checks string length
    // Space not in the substring, numbers, hiragana and more should be taken
    // care of within pre_tokenizers.
    // https://github.com/google/sentencepiece/blob/26be9516cd81d5315ee31c48d2438018e0eab879/src/trainer_interface.cc#L203
    let n = char_string.len();
    if char_string.is_empty() || n > max_piece_length {
        return false;
    }

    true
}

/// Build the seed vocabulary shared by Unigram-family trainers.
pub(crate) fn make_seed_pieces(
    sentences: &[Sentence],
    seed_init: SeedInit,
    seed_size: usize,
    max_piece_length: usize,
    initial_alphabet: &AHashSet<char>,
    progress: &Option<ProgressBar>,
) -> Result<Vec<SentencePiece>> {
    match seed_init {
        SeedInit::Substrings => Ok(make_substring_seed_pieces(
            sentences,
            seed_size,
            max_piece_length,
            progress,
        )),
        SeedInit::Bpe => {
            make_bpe_seed_pieces(sentences, seed_size, max_piece_length, initial_alphabet)
        }
    }
}

fn make_substring_seed_pieces(
    sentences: &[Sentence],
    seed_size: usize,
    max_piece_length: usize,
    _progress: &Option<ProgressBar>,
) -> Vec<SentencePiece> {
    // Put all sentences in a string, separated by \0
    let total: usize = sentences
        .iter()
        .map(|(s, _)| s.chars().count())
        .sum::<usize>()
        + sentences.len();
    let mut flat_string = String::with_capacity(total);
    let mut all_chars: AHashMap<char, u32> = AHashMap::new();
    let c_sentence_boundary = '\0';
    let k_sentence_boundary = '\0'.to_string();
    for (string, n) in sentences {
        if string.is_empty() {
            continue;
        }
        flat_string.push_str(string);
        // XXX
        // Comment suggests we add sentence boundary, but it seems to be missing from actual
        // code in spm.
        flat_string.push_str(&k_sentence_boundary);
        for c in string.chars() {
            if c != c_sentence_boundary {
                *all_chars.entry(c).or_insert(0) += n;
            }
        }
    }
    flat_string.shrink_to_fit();
    #[cfg(feature = "esaxx_fast")]
    let suffix = esaxx_rs::suffix(&flat_string).unwrap();
    #[cfg(not(feature = "esaxx_fast"))]
    let suffix = esaxx_rs::suffix_rs(&flat_string).unwrap();

    //  Basic chars need to be in sentence pieces.
    let mut seed_sentencepieces: Vec<SentencePiece> = vec![];

    let mut sall_chars: Vec<_> = all_chars.into_iter().map(|(a, b)| (b, a)).collect();
    // Reversed order
    sall_chars.sort_by_key(|&a| Reverse(a));
    let mut substr_index: Vec<_> = suffix
        .iter()
        .filter_map(|(string, freq)| {
            if string.len() <= 1 {
                return None;
            }
            if string.contains(&c_sentence_boundary) {
                return None;
            }
            if !is_valid_sentencepiece(string, max_piece_length) {
                return None;
            }
            let score = freq * string.len() as u32;
            // if let Some(p) = &progress {
            //     p.inc(1);
            // }
            Some((score, string))
        })
        .collect();

    // Fill seed_sentencepieces
    for (count, character) in sall_chars {
        seed_sentencepieces.push((character.to_string(), count.into()));
    }

    // sort by decreasing score
    substr_index.sort_by_key(|&a| Reverse(a));
    for (score, char_string) in substr_index {
        // Just in case
        assert!(is_valid_sentencepiece(char_string, max_piece_length));
        let string: String = char_string.iter().collect();
        seed_sentencepieces.push((string, score.into()));
        if seed_sentencepieces.len() >= seed_size {
            break;
        }
    }
    to_log_prob(&mut seed_sentencepieces);
    seed_sentencepieces
}

#[cfg(feature = "bpe")]
fn make_bpe_seed_pieces(
    sentences: &[Sentence],
    seed_size: usize,
    max_piece_length: usize,
    initial_alphabet: &AHashSet<char>,
) -> Result<Vec<SentencePiece>> {
    // Deterministic input order: BPE counts are order-independent integers,
    // but the re-segmentation counts below sum floats per piece.
    let mut ordered: Vec<&Sentence> = sentences.iter().collect();
    ordered.sort_by(|a, b| a.0.cmp(&b.0));
    let word_counts: AHashMap<CompactString, u64> = ordered
        .iter()
        .map(|(word, count)| (CompactString::from(word), *count as u64))
        .collect();

    // BPE's max_token_length counts bytes; every char is at most 4 bytes, so
    // this bound excludes no piece that the char-based filter below would keep.
    let bpe_trainer = BpeTrainer::builder()
        .vocab_size(seed_size)
        .show_progress(false)
        .max_token_length(Some(max_piece_length * 4))
        .initial_alphabet(initial_alphabet.iter().copied().collect())
        .build();
    let (vocab, merges, _special) = bpe_trainer.do_train(&word_counts)?;
    // Every BPE token seeds, in vocab order. Tokens the re-segmentation never
    // emits (a single always covered by longer merges, intermediate merges)
    // floor at count 1 rather than vanishing.
    let mut by_id: Vec<(u32, String)> = vocab
        .iter()
        .map(|(piece, id)| (*id, piece.clone()))
        .collect();
    by_id.sort_by_key(|(id, _)| *id);
    let bpe = PipelineBPE::from_config(BpeConfig {
        vocab,
        merges,
        ..BpeConfig::default()
    })?;

    let mut scratch = bpe.init_scratch();
    let mut counts: AHashMap<u32, f64> = AHashMap::new();
    for (word, freq) in ordered {
        if word.is_empty() {
            continue;
        }
        let mut output = vec![];
        bpe.tokenize_pipeline(word, &mut scratch, &mut output)?;
        for token in output {
            *counts.entry(token.id()).or_insert(0.0) += *freq as f64;
        }
    }
    let mut seed: Vec<SentencePiece> = by_id
        .into_iter()
        .filter(|(_, piece)| {
            is_valid_sentencepiece(&piece.chars().collect::<Vec<_>>(), max_piece_length)
        })
        .map(|(id, piece)| {
            let count = counts.get(&id).copied().unwrap_or(0.0).max(1.0);
            (piece, count)
        })
        .collect();
    to_log_prob(&mut seed);
    Ok(seed)
}

#[cfg(not(feature = "bpe"))]
fn make_bpe_seed_pieces(
    _sentences: &[Sentence],
    _seed_size: usize,
    _max_piece_length: usize,
    _initial_alphabet: &AHashSet<char>,
) -> Result<Vec<SentencePiece>> {
    Err("seed_init=\"bpe\" needs tk-train's `bpe` feature".into())
}

/// Characters that must survive training: every char seen in the words plus
/// the initial alphabet.
pub(crate) fn required_chars(
    word_counts: &[Sentence],
    initial_alphabet: &AHashSet<char>,
) -> AHashSet<String> {
    word_counts
        .iter()
        .flat_map(|(s, _count)| s.chars())
        .chain(initial_alphabet.iter().copied())
        .map(|c| c.to_string())
        .collect()
}

/// Proper UTF-8 prefixes of observed multibyte chars (`<0xE0A4>` style):
/// the multibyte half of the byte floor. Only chars of 3+ bytes contribute;
/// 2-byte chars are already covered by the singles. Sorted for determinism.
fn prefix_pieces(required_chars: &AHashSet<String>) -> Vec<String> {
    let mut prefixes: Vec<String> = vec![];
    for c in required_chars {
        let bytes = c.as_bytes();
        if bytes.len() >= 3 {
            for len in 2..bytes.len() {
                let hex: String = bytes[..len].iter().map(|b| format!("{b:02X}")).collect();
                prefixes.push(format!("<0x{hex}>"));
            }
        }
    }
    prefixes.sort();
    prefixes.dedup();
    prefixes
}

/// Lay out the final model: required chars first (reusing trained scores,
/// backfilling missing chars just below the minimum), truncated to
/// `vocab_size` including special tokens and the unk token. With
/// `byte_prefixes`, the 256 single-byte pieces plus observed multibyte
/// prefixes are appended after the truncation, so the learned vocabulary is
/// unchanged and enabling the option only appends.
pub(crate) fn finalize(
    vocab_size: u32,
    special_tokens: &[AddedToken],
    unk_token: &Option<String>,
    model: &Unigram,
    required_chars: AHashSet<String>,
    byte_prefixes: bool,
) -> Result<Unigram> {
    let mut min_score_penalty = 0.0;
    let min_score_penalty_delta = 0.0001;

    let byte_candidates: Vec<String> = if byte_prefixes {
        let mut out: Vec<String> = (0..=255u32).map(|b| format!("<0x{b:02X}>")).collect();
        out.extend(prefix_pieces(&required_chars));
        out
    } else {
        vec![]
    };

    let mut pieces: Vec<(String, f64)> = vec![];
    let mut inserted: AHashSet<String> = AHashSet::new();

    // We don't want to include the <UNK> that was used to train
    inserted.insert("<UNK>".into());

    let existing_pieces: AHashMap<String, f64> = model.iter().cloned().collect();
    // Deterministic order: backfill scores and stable-sort ties must not
    // depend on hash iteration order.
    let mut required_chars: Vec<String> = required_chars.into_iter().collect();
    required_chars.sort();
    for c in required_chars {
        if let Some(t) = existing_pieces.get(&c) {
            inserted.insert(c.clone());
            pieces.push((c, *t));
        } else {
            let score = model.min_score + min_score_penalty;

            inserted.insert(c.clone());
            pieces.push((c, score));
            min_score_penalty += min_score_penalty_delta;
        }
    }

    let (unk_id, need_add_unk) = if let Some(unk) = unk_token {
        let unk_id = special_tokens
            .iter()
            .enumerate()
            .find_map(|(i, t)| if t.content == *unk { Some(i) } else { None });
        match unk_id {
            Some(id) => (Some(id), false),
            None => (Some(0), true),
        }
    } else {
        (None, false)
    };

    let vocab_size_without_special_tokens = if need_add_unk {
        vocab_size as usize - special_tokens.len() - 1
    } else {
        vocab_size as usize - special_tokens.len()
    };
    for (token, score) in model.iter() {
        if inserted.contains::<str>(token) {
            continue;
        }
        inserted.insert(token.to_string());
        pieces.push((token.to_string(), if score.is_nan() { 0.0 } else { *score }));

        if pieces.len() == vocab_size_without_special_tokens {
            break;
        }
    }
    pieces.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
    for spelling in byte_candidates {
        if !inserted.contains::<str>(&spelling) {
            pieces.push((spelling, 0.0));
        }
    }

    // Insert the necessary tokens
    let mut special_tokens = special_tokens
        .iter()
        .map(|t| (t.content.clone(), 0.0))
        .collect::<Vec<_>>();
    if need_add_unk {
        special_tokens.insert(0, (unk_token.clone().unwrap(), 0.0));
    }

    Unigram::from(
        special_tokens.into_iter().chain(pieces).collect(),
        unk_id,
        model.byte_fallback(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_substring_seed_chars() {
        let sentences = vec![
            ("This is a".to_string(), 1),
            ("こんにちは友達".to_string(), 1),
        ];

        let table = make_substring_seed_pieces(&sentences, 1_000_000, 16, &None);

        let target_strings = vec![
            "s", "i", " ", "達", "友", "ん", "は", "に", "ち", "こ", "h", "a", "T", "is ", "s ",
        ];

        let strings: Vec<_> = table.iter().map(|(string, _)| string).collect();
        assert_eq!(strings, target_strings);

        let scores = table.iter().map(|(_, score)| score);
        let target_scores = vec![
            -2.5649493574615367, // 2.0
            -2.5649493574615367, // 2.0
            -2.5649493574615367, // 2.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -1.4663370687934272, // 6.0
            -1.8718021769015916, // 4.0
        ];

        for (score, target_score) in scores.zip(target_scores) {
            assert_approx_eq!(*score, target_score, 0.01);
        }
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn test_bpe_seed_pieces() {
        let sentences = vec![
            ("ab".to_string(), 10),
            ("aab".to_string(), 5),
            ("b".to_string(), 3),
        ];
        let alphabet = AHashSet::new();

        let seed = make_bpe_seed_pieces(&sentences, 10, 16, &alphabet).unwrap();

        // Single chars are always in the seed, longer BPE tokens join them,
        // everything within max_piece_length, scores are log-probs.
        let pieces: Vec<&str> = seed.iter().map(|(s, _)| s.as_str()).collect();
        assert!(pieces.contains(&"a"));
        assert!(pieces.contains(&"b"));
        assert!(pieces.iter().all(|s| s.chars().count() <= 16));
        let total: f64 = seed.iter().map(|(_, score)| score.exp()).sum();
        assert_approx_eq!(total, 1.0, 1e-9);
        // No duplicates.
        let mut dedup = pieces.clone();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(dedup.len(), pieces.len());
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn test_bpe_seed_respects_max_piece_length() {
        let sentences = vec![("abcdefgh".to_string(), 20)];
        let alphabet = AHashSet::new();

        let seed = make_bpe_seed_pieces(&sentences, 20, 3, &alphabet).unwrap();

        assert!(!seed.is_empty());
        assert!(seed.iter().all(|(s, _)| s.chars().count() <= 3));
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn test_bpe_seed_is_deterministic() {
        let sentences = vec![
            ("hello".to_string(), 7),
            ("world".to_string(), 5),
            ("hello world".to_string(), 3),
            ("low".to_string(), 4),
        ];
        let alphabet = AHashSet::new();

        let first = make_bpe_seed_pieces(&sentences, 30, 16, &alphabet).unwrap();
        let second = make_bpe_seed_pieces(&sentences, 30, 16, &alphabet).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn test_required_chars() {
        use std::iter::FromIterator;
        let sentences = vec![("こんにちは友達".to_string(), 1)];
        let alphabet = AHashSet::from_iter(vec!['a', 'b', 'c', 'd', 'e', 'f']);
        assert_eq!(
            required_chars(&sentences, &alphabet),
            vec![
                "こ", "ん", "に", "ち", "は", "友", "達", "a", "b", "c", "d", "e", "f"
            ]
            .into_iter()
            .map(|s| s.to_owned())
            .collect::<AHashSet<_>>()
        );
    }

    #[test]
    fn test_to_log_prob() {
        let mut a = vec![("".to_string(), 1.0), ("".to_string(), 2.0)];
        to_log_prob(&mut a);
        let scores = a.iter().map(|(_, score)| *score).collect::<Vec<_>>();
        // ln(1) - ln(3)
        assert_approx_eq!(scores[0], -1.098, 0.01);
        // ln(2) - ln(3)
        assert_approx_eq!(scores[1], -0.405, 0.01);
    }
}
