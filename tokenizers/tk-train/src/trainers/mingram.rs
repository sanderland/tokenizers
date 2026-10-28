//! MinGram ("Minimalist Unigram") trainer: BPE-seeded hard-EM training that
//! outputs a standard `Unigram` model with `objective = min_tokens`.
//!
//! Seed -> hard EM (minimum-token-count E-step, MLE M-step) -> one flat
//! score prune -> shared finalize. No forward-backward pass, no iterative
//! prune loop.

use super::seed::{self, SeedInit, Sentence, SentencePiece};
use super::unigram::UnigramTrainerError;
use crate::Trainer;
use crate::progress::{ProgressBar, ProgressStyle};
use ahash::{AHashMap, AHashSet};
use log::debug;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::convert::TryInto;
use tk_encode::Result;
use tk_encode::models::unigram::{
    lattice::Lattice,
    model::{Objective, Unigram},
};
use tk_encode::utils::parallelism::*;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

/// A `MinGramTrainer` trains a `Unigram` model from `word_counts` with hard EM.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinGramTrainer {
    pub show_progress: bool,
    pub vocab_size: u32,
    pub n_sub_iterations: u32,
    #[serde(with = "crate::added_token_serde")]
    pub special_tokens: Vec<AddedToken>,
    pub initial_alphabet: AHashSet<char>,

    pub unk_token: Option<String>,

    pub max_piece_length: usize,
    seed_size: usize,
    pub seed_init: SeedInit,
    pub byte_prefixes: bool,
    words: AHashMap<String, u32>,
}

/// Builds a [`MinGramTrainer`]. Every field has a default, so `build` cannot fail -- the
/// [`Infallible`] error is kept only so callers can go on writing `.build().unwrap()`.
#[derive(Debug, Clone, Default)]
pub struct MinGramTrainerBuilder {
    show_progress: Option<bool>,
    vocab_size: Option<u32>,
    n_sub_iterations: Option<u32>,
    special_tokens: Option<Vec<AddedToken>>,
    initial_alphabet: Option<AHashSet<char>>,
    unk_token: Option<Option<String>>,
    max_piece_length: Option<usize>,
    seed_size: Option<usize>,
    seed_init: Option<SeedInit>,
    byte_prefixes: Option<bool>,
    words: Option<AHashMap<String, u32>>,
}

impl MinGramTrainerBuilder {
    pub fn show_progress(&mut self, show_progress: bool) -> &mut Self {
        self.show_progress = Some(show_progress);
        self
    }

    pub fn vocab_size(&mut self, vocab_size: u32) -> &mut Self {
        self.vocab_size = Some(vocab_size);
        self
    }

    pub fn n_sub_iterations(&mut self, n_sub_iterations: u32) -> &mut Self {
        self.n_sub_iterations = Some(n_sub_iterations);
        self
    }

    pub fn special_tokens(&mut self, special_tokens: Vec<AddedToken>) -> &mut Self {
        self.special_tokens = Some(special_tokens);
        self
    }

    pub fn initial_alphabet(&mut self, initial_alphabet: AHashSet<char>) -> &mut Self {
        self.initial_alphabet = Some(initial_alphabet);
        self
    }

    pub fn unk_token(&mut self, unk_token: Option<String>) -> &mut Self {
        self.unk_token = Some(unk_token);
        self
    }

    pub fn max_piece_length(&mut self, max_piece_length: usize) -> &mut Self {
        self.max_piece_length = Some(max_piece_length);
        self
    }

    pub fn seed_size(&mut self, seed_size: usize) -> &mut Self {
        self.seed_size = Some(seed_size);
        self
    }

    pub fn seed_init(&mut self, seed_init: SeedInit) -> &mut Self {
        self.seed_init = Some(seed_init);
        self
    }

    /// Emit `<0x..>` byte pieces (256 singles plus observed multibyte
    /// prefixes) so inference can tile uncovered bytes: ctok's byte floor.
    pub fn byte_prefixes(&mut self, byte_prefixes: bool) -> &mut Self {
        self.byte_prefixes = Some(byte_prefixes);
        self
    }

    pub fn words(&mut self, words: AHashMap<String, u32>) -> &mut Self {
        self.words = Some(words);
        self
    }

    pub fn build(&self) -> std::result::Result<MinGramTrainer, Infallible> {
        Ok(MinGramTrainer {
            show_progress: self.show_progress.unwrap_or(true),
            vocab_size: self.vocab_size.unwrap_or(8000),
            n_sub_iterations: self.n_sub_iterations.unwrap_or(2),
            special_tokens: self.special_tokens.clone().unwrap_or_default(),
            initial_alphabet: self.initial_alphabet.clone().unwrap_or_default(),
            unk_token: self.unk_token.clone().unwrap_or(None),
            max_piece_length: self.max_piece_length.unwrap_or(16),
            seed_size: self.seed_size.unwrap_or(1_000_000),
            seed_init: self.seed_init.unwrap_or(SeedInit::Bpe),
            byte_prefixes: self.byte_prefixes.unwrap_or(false),
            words: self.words.clone().unwrap_or_default(),
        })
    }
}

impl Default for MinGramTrainer {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

/// Keep the training UNK placeholder, all single chars, and the top-scoring
/// longer pieces up to `target_size`. Stable: score ties keep seed order.
fn flat_prune(pieces: &[SentencePiece], target_size: usize) -> Vec<SentencePiece> {
    if pieces.len() <= target_size {
        return pieces.to_vec();
    }
    let mut kept = vec![pieces[0].clone()];
    kept.extend(
        pieces[1..]
            .iter()
            .filter(|(piece, _)| piece.chars().count() == 1)
            .cloned(),
    );
    let mut rest: Vec<&SentencePiece> = pieces[1..]
        .iter()
        .filter(|(piece, _)| piece.chars().count() != 1)
        .collect();
    rest.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
    for (piece, score) in rest {
        if kept.len() >= target_size {
            break;
        }
        kept.push((piece.clone(), *score));
    }
    kept
}

impl MinGramTrainer {
    pub fn builder() -> MinGramTrainerBuilder {
        MinGramTrainerBuilder::default()
    }

    /// Setup a progress bar if asked to show progress
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            Some(p)
        } else {
            None
        }
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
    }

    /// Set the progress bar in the finish state
    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
    }

    /// Hard E-step: segment each word on its minimum-token-count path and
    /// accumulate frequency-weighted piece counts. No forward-backward pass.
    fn run_e_step(&self, model: &Unigram, sentences: &[Sentence]) -> (u32, Vec<f64>) {
        let chunk_size = std::cmp::max(sentences.len() / current_num_threads(), 1);
        sentences
            .maybe_par_chunks(chunk_size)
            .map(|sentences_chunk| {
                let mut expected: Vec<f64> = vec![0.0; model.len()];
                let mut ntokens: u32 = 0;

                for (string, freq) in sentences_chunk {
                    let mut lattice = Lattice::from(string, model.bos_id, model.eos_id);
                    model.populate_nodes(&mut lattice);

                    let path = lattice.viterbi_min_tokens();
                    for node_ref in &path {
                        let id = node_ref.borrow().id;
                        expected[id] += *freq as f64;
                    }
                    ntokens += path.len() as u32;
                }
                (ntokens, expected)
            })
            .reduce(
                || (0, vec![0.0; model.len()]),
                |(ntokens, expected), (lntokens, lexpected)| {
                    (
                        ntokens + lntokens,
                        expected
                            .iter()
                            .zip(lexpected)
                            .map(|(global_el, local_el)| global_el + local_el)
                            .collect(),
                    )
                },
            )
    }

    /// M-step: drop pieces with negligible expected counts and re-estimate
    /// plain MLE log-probs (no Bayesian prior).
    fn run_m_step(&self, pieces: &[SentencePiece], expected: &[f64]) -> Vec<SentencePiece> {
        if pieces.len() != expected.len() {
            panic!(
                "Those two iterators are supposed to be the same length ({} vs {})",
                pieces.len(),
                expected.len()
            );
        }
        let mut new_pieces: Vec<SentencePiece> =
            Vec::with_capacity(self.vocab_size.try_into().unwrap());

        let mut sum = 0.0;
        let expected_frequency_threshold = 0.5;

        for (i, (freq, (piece, _score))) in expected.iter().zip(pieces).enumerate() {
            // Always keep unk.
            if i == 0 {
                new_pieces.push((piece.clone(), f64::NAN));
                continue;
            }
            if *freq < expected_frequency_threshold {
                continue;
            }
            new_pieces.push((piece.clone(), *freq));
            sum += freq;
        }
        let logsum = sum.ln();
        new_pieces
            .into_iter()
            .map(|(s, c)| (s, c.ln() - logsum))
            .collect()
    }

    /// Re-estimate MLE log-probs for a fixed piece list by hard EM: segment
    /// the sentences on min-token paths and normalize the counts. Nothing is
    /// pruned, so the output aligns with the input pieces; unseen pieces floor
    /// at 0.5, below any seen count, which keeps their scores finite (the
    /// `tokenizer.json` writer rejects non-finite scores) and losing every tie.
    pub fn em_scores(
        &self,
        pieces: &[String],
        sentences: &[(String, u32)],
    ) -> Result<Vec<(String, f64)>> {
        let mut model_pieces: Vec<SentencePiece> = std::iter::once(("<UNK>".into(), f64::NAN))
            .chain(pieces.iter().map(|piece| (piece.clone(), 0.0)))
            .collect();
        let mut new_model = Unigram::from(model_pieces.clone(), Some(0), false)?;
        for _ in 0..self.n_sub_iterations {
            let (_, expected) = self.run_e_step(&new_model, sentences);
            let total: f64 = expected[1..].iter().map(|count| count.max(0.5)).sum();
            let logsum = total.ln();
            model_pieces = std::iter::once(("<UNK>".into(), f64::NAN))
                .chain(
                    pieces
                        .iter()
                        .zip(&expected[1..])
                        .map(|(piece, count)| (piece.clone(), count.max(0.5).ln() - logsum)),
                )
                .collect();
            new_model = Unigram::from(model_pieces.clone(), Some(0), false)?;
        }
        Ok(model_pieces.into_iter().skip(1).collect())
    }

    pub fn do_train(
        &self,
        sentences: Vec<Sentence>,
        model: &mut Unigram,
    ) -> Result<Vec<AddedToken>> {
        let progress = self.setup_progress();
        //
        // 1. Seed the vocabulary (BPE tokens by default)
        self.update_progress(&progress, sentences.len(), "Seeding vocabulary");
        let mut pieces: Vec<SentencePiece> =
            Vec::with_capacity(self.vocab_size.try_into().unwrap());

        // We use a UNK token when training, whatever the `self.unk_token`
        pieces.push(("<UNK>".into(), f64::NAN));
        pieces.extend(seed::make_seed_pieces(
            &sentences,
            self.seed_init,
            self.seed_size,
            self.max_piece_length,
            &self.initial_alphabet,
            &progress,
        )?);
        self.finalize_progress(&progress, sentences.len());

        debug!(
            "Using {} pieces on {} sentences for hard EM training",
            pieces.len(),
            sentences.len()
        );

        let required_chars = seed::required_chars(&sentences, &self.initial_alphabet);
        if required_chars.len() as u32 > self.vocab_size {
            return Err(Box::new(UnigramTrainerError::VocabularyTooSmall));
        }

        // 2. Run hard EM.
        let expected_updates = self.n_sub_iterations as usize;
        self.update_progress(&progress, expected_updates, "Hard EM training");
        let mut new_model = Unigram::from(pieces.clone(), Some(0), false)?;
        for _iter in 0..self.n_sub_iterations {
            let (num_tokens, expected) = self.run_e_step(&new_model, &sentences);
            pieces = self.run_m_step(&pieces, &expected);
            new_model = Unigram::from(pieces.clone(), Some(0), false)?;

            debug!(
                "Hard EM iter={} size={} num_tokens={}",
                _iter,
                new_model.len(),
                num_tokens,
            );
            if let Some(p) = &progress {
                p.inc(1);
            }
        }
        self.finalize_progress(&progress, expected_updates);

        // 3. One flat score prune down to `vocab_size`.
        pieces = flat_prune(&pieces, self.vocab_size as usize);
        new_model = Unigram::from(pieces.clone(), Some(0), false)?;

        // 4. Adjust the size of sentencepieces to be |vocab_size|.
        *model = seed::finalize(
            self.vocab_size,
            &self.special_tokens,
            &self.unk_token,
            &new_model,
            required_chars,
            self.byte_prefixes,
        )?;
        model.set_objective(Objective::MinTokens);

        Ok(self.special_tokens.clone())
    }
}

impl Trainer for MinGramTrainer {
    type Model = Unigram;

    /// Train a Unigram model
    fn train(&self, model: &mut Unigram) -> Result<Vec<AddedToken>> {
        let sentences: Vec<_> = self.words.iter().map(|(s, i)| (s.to_owned(), *i)).collect();
        self.do_train(sentences, model)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<AHashMap<String, u32>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = AHashMap::new();
                for word in words {
                    *map.entry(word).or_default() += 1;
                }
                Ok(map)
            })
            .reduce(
                || Ok(AHashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        *acc.entry(k).or_default() += v;
                    }
                    Ok(acc)
                },
            );

        self.words = words?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    fn tiny_corpus() -> Vec<Sentence> {
        vec![
            ("hello".to_string(), 10),
            ("world".to_string(), 8),
            ("hello world".to_string(), 3),
            ("low".to_string(), 5),
            ("lower".to_string(), 2),
        ]
    }

    #[test]
    fn test_hard_e_step_counts_min_paths() {
        let pieces = vec![
            ("<UNK>".to_string(), f64::NAN),
            ("a".to_string(), -1.0),
            ("b".to_string(), -1.0),
            ("ab".to_string(), -10.0),
        ];
        let model = Unigram::from(pieces, Some(0), false).unwrap();
        let trainer = MinGramTrainer::builder().build().unwrap();

        // "ab" scores below [a, b], but the min-count path still takes it.
        let (ntokens, expected) = trainer.run_e_step(&model, &[("ab".to_string(), 3)]);
        assert_eq!(ntokens, 1);
        assert_eq!(expected, vec![0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_m_step_drops_zero_counts_and_reestimates() {
        let trainer = MinGramTrainer::builder().build().unwrap();
        let pieces = vec![
            ("<UNK>".to_string(), f64::NAN),
            ("a".to_string(), -1.0),
            ("b".to_string(), -1.0),
            ("ab".to_string(), -1.0),
        ];
        let new_pieces = trainer.run_m_step(&pieces, &[0.0, 4.0, 0.0, 2.0]);

        assert_eq!(new_pieces.len(), 3);
        assert_eq!(new_pieces[0].0, "<UNK>");
        assert!(new_pieces[0].1.is_nan());
        assert_eq!(new_pieces[1].0, "a");
        assert_approx_eq!(new_pieces[1].1, (4.0_f64 / 6.0).ln(), 1e-9);
        assert_eq!(new_pieces[2].0, "ab");
        assert_approx_eq!(new_pieces[2].1, (2.0_f64 / 6.0).ln(), 1e-9);
    }

    #[test]
    fn test_flat_prune_keeps_unk_singles_and_top_scores() {
        let pieces = vec![
            ("<UNK>".to_string(), f64::NAN),
            ("a".to_string(), -9.0),
            ("b".to_string(), -9.0),
            ("ab".to_string(), -1.0),
            ("bc".to_string(), -5.0),
            ("abc".to_string(), -2.0),
        ];
        let pruned = flat_prune(&pieces, 4);

        let strings: Vec<&str> = pruned.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(strings, vec!["<UNK>", "a", "b", "ab"]);
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn test_mingram_trains_with_bpe_seed() {
        let trainer = MinGramTrainer::builder()
            .show_progress(false)
            .vocab_size(20)
            .seed_size(30)
            .build()
            .unwrap();
        assert_eq!(trainer.seed_init, SeedInit::Bpe);

        let mut model = Unigram::default();
        trainer.do_train(tiny_corpus(), &mut model).unwrap();

        assert_eq!(model.objective(), Objective::MinTokens);
        assert!(model.get_vocab_size() <= 20);
        // Every corpus char is encodable.
        for c in ["h", "e", "l", "o", " ", "w", "r", "d"] {
            assert!(model.token_to_id(c).is_some(), "missing char {}", c);
        }
        // Encoding works and uses multi-char pieces.
        let encoded = model.encode("hello").unwrap();
        assert!(encoded.len() < "hello".len());
    }

    #[test]
    fn test_mingram_trains_with_substring_seed() {
        let trainer = MinGramTrainer::builder()
            .show_progress(false)
            .vocab_size(20)
            .seed_init(SeedInit::Substrings)
            .build()
            .unwrap();

        let mut model = Unigram::default();
        trainer.do_train(tiny_corpus(), &mut model).unwrap();

        assert_eq!(model.objective(), Objective::MinTokens);
        assert!(model.get_vocab_size() <= 20);
    }

    #[test]
    fn test_em_scores_keep_vocab_and_estimate_mle() {
        let trainer = MinGramTrainer::builder()
            .n_sub_iterations(3)
            .build()
            .unwrap();
        let pieces = ["a", "b", "ab"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let sentences = vec![("ab".to_string(), 3), ("a".to_string(), 1)];

        let scored = trainer.em_scores(&pieces, &sentences).unwrap();

        // Nothing pruned: output aligns with input.
        assert_eq!(
            scored.iter().map(|(p, _)| p).collect::<Vec<_>>(),
            ["a", "b", "ab"]
        );
        // Min-count paths use "ab" x3 and "a" x1; "b" is unseen (floor 0.5).
        let total = 1.0 + 0.5 + 3.0;
        assert_approx_eq!(scored[0].1, (1.0_f64 / total).ln(), 1e-9);
        assert_approx_eq!(scored[1].1, (0.5_f64 / total).ln(), 1e-9);
        assert_approx_eq!(scored[2].1, (3.0_f64 / total).ln(), 1e-9);
        assert!(scored.iter().all(|(_, s)| s.is_finite()));
    }

    #[test]
    fn test_mingram_output_encodes_under_max_score() {
        // Cross-combination: min_tokens never uses more pieces than max_score.
        let trainer = MinGramTrainer::builder()
            .show_progress(false)
            .vocab_size(20)
            .seed_init(SeedInit::Substrings)
            .build()
            .unwrap();

        let mut model = Unigram::default();
        trainer.do_train(tiny_corpus(), &mut model).unwrap();

        let min_count = model.encode("hello").unwrap().len();
        model.set_objective(Objective::MaxScore);
        let max_count = model.encode("hello").unwrap().len();
        assert!(min_count <= max_count);
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn test_mingram_is_deterministic() {
        let train = || {
            let trainer = MinGramTrainer::builder()
                .show_progress(false)
                .vocab_size(20)
                .seed_size(30)
                .build()
                .unwrap();
            let mut model = Unigram::default();
            trainer.do_train(tiny_corpus(), &mut model).unwrap();
            model.iter().cloned().collect::<Vec<_>>()
        };
        assert_eq!(train(), train());
    }

    #[test]
    fn test_byte_prefixes_opt_in() {
        let corpus = || -> Vec<Sentence> {
            vec![("hello".into(), 10), ("日本語".into(), 8), ("🎉".into(), 3)]
        };
        let train = |byte_prefixes: bool| {
            let trainer = MinGramTrainer::builder()
                .show_progress(false)
                .vocab_size(30)
                .seed_size(40)
                .seed_init(SeedInit::Substrings)
                .byte_prefixes(byte_prefixes)
                .build()
                .unwrap();
            let mut model = Unigram::default();
            trainer.do_train(corpus(), &mut model).unwrap();
            model.iter().cloned().collect::<Vec<_>>()
        };
        let off = train(false);
        assert!(!off.iter().any(|(t, _)| t == "<0x00>"));
        let on = train(true);
        // The learned vocabulary is unchanged; the option only appends.
        assert_eq!(&on[..off.len()], &off[..]);
        let spellings: Vec<&str> = on.iter().map(|(t, _)| t.as_str()).collect();
        // 日 E697A5 -> E697; 🎉 F09F8E89 -> F09F, F09F8E.
        for piece in ["<0x00>", "<0xFF>", "<0xE697>", "<0xF09F>", "<0xF09F8E>"] {
            assert!(spellings.contains(&piece), "missing {piece}");
        }
        assert_eq!(train(true), on);
    }
}
