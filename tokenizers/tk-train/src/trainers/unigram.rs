use super::seed::{self, SeedInit, Sentence, SentencePiece};
use crate::Trainer;
use crate::progress::{ProgressBar, ProgressStyle};
use ahash::{AHashMap, AHashSet};
use log::debug;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::convert::TryInto;
use tk_encode::Result;
use tk_encode::models::unigram::{lattice::Lattice, model::Unigram};
use tk_encode::utils::parallelism::*;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 7.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    x -= 1.0 / 2.0;
    let xx = 1.0 / x;
    let xx2 = xx * xx;
    let xx4 = xx2 * xx2;
    result += x.ln() + (1.0 / 24.0) * xx2 - 7.0 / 960.0 * xx4 + (31.0 / 8064.0) * xx4 * xx2
        - (127.0 / 30720.0) * xx4 * xx4;
    result
}

#[derive(thiserror::Error, Debug)]
pub enum UnigramTrainerError {
    #[error("The vocabulary is not large enough to contain all chars")]
    VocabularyTooSmall,
}

/// A `UnigramTrainer` can train a `Unigram` model from `word_counts`.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnigramTrainer {
    pub show_progress: bool,
    pub vocab_size: u32,
    pub n_sub_iterations: u32,
    pub shrinking_factor: f64,
    #[serde(with = "crate::added_token_serde")]
    pub special_tokens: Vec<AddedToken>,
    pub initial_alphabet: AHashSet<char>,

    pub unk_token: Option<String>,

    pub max_piece_length: usize,
    seed_size: Option<usize>,
    pub seed_init: SeedInit,
    words: AHashMap<String, u32>,
}

/// Builds a [`UnigramTrainer`]. Every field has a default, so `build` cannot fail -- the
/// [`Infallible`] error is kept only so callers can go on writing `.build().unwrap()`.
#[derive(Debug, Clone, Default)]
pub struct UnigramTrainerBuilder {
    show_progress: Option<bool>,
    vocab_size: Option<u32>,
    n_sub_iterations: Option<u32>,
    shrinking_factor: Option<f64>,
    special_tokens: Option<Vec<AddedToken>>,
    initial_alphabet: Option<AHashSet<char>>,
    unk_token: Option<Option<String>>,
    max_piece_length: Option<usize>,
    seed_size: Option<usize>,
    seed_init: Option<SeedInit>,
    words: Option<AHashMap<String, u32>>,
}

impl UnigramTrainerBuilder {
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

    pub fn shrinking_factor(&mut self, shrinking_factor: f64) -> &mut Self {
        self.shrinking_factor = Some(shrinking_factor);
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

    pub fn words(&mut self, words: AHashMap<String, u32>) -> &mut Self {
        self.words = Some(words);
        self
    }

    pub fn build(&self) -> std::result::Result<UnigramTrainer, Infallible> {
        Ok(UnigramTrainer {
            show_progress: self.show_progress.unwrap_or(true),
            vocab_size: self.vocab_size.unwrap_or(8000),
            n_sub_iterations: self.n_sub_iterations.unwrap_or(2),
            shrinking_factor: self.shrinking_factor.unwrap_or(0.75),
            special_tokens: self.special_tokens.clone().unwrap_or_default(),
            initial_alphabet: self.initial_alphabet.clone().unwrap_or_default(),
            unk_token: self.unk_token.clone().unwrap_or(None),
            max_piece_length: self.max_piece_length.unwrap_or(16),
            seed_size: self.seed_size,
            seed_init: self.seed_init.unwrap_or_default(),
            words: self.words.clone().unwrap_or_default(),
        })
    }
}

impl Default for UnigramTrainer {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

impl UnigramTrainer {
    pub fn builder() -> UnigramTrainerBuilder {
        UnigramTrainerBuilder::default()
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

    fn prune_sentence_pieces(
        &self,
        model: &Unigram,
        pieces: &[SentencePiece],
        sentences: &[Sentence],
    ) -> Vec<SentencePiece> {
        let mut always_keep = vec![true; pieces.len()];
        let mut alternatives: Vec<Vec<usize>> = vec![Vec::new(); pieces.len()];

        let bos_id = pieces.len() + 1;
        let eos_id = pieces.len() + 2;

        // First, segments the current sentencepieces to know
        // how each sentencepiece is resegmented if this sentencepiece is removed
        // from the vocabulary.
        // To do so, we take the second best segmentation of sentencepiece[i].
        // alternatives[i] stores the sequence of second best sentencepieces.
        for (id, (token, _score)) in pieces.iter().enumerate() {
            // Always keep unk.
            if id == 0 {
                always_keep[id] = false;
                continue;
            }
            let mut lattice = Lattice::from(token, bos_id, eos_id);
            model.populate_nodes(&mut lattice);

            let nbests = lattice.nbest(2);
            if nbests.len() == 1 {
                always_keep[id] = true;
            } else if nbests[0].len() >= 2 {
                always_keep[id] = false;
            } else if nbests[0].len() == 1 {
                always_keep[id] = true;
                for node in &nbests[1] {
                    let alt_id = node.borrow().id;
                    alternatives[id].push(alt_id);
                }
            }
        }

        // Second, segments all sentences to compute likelihood
        // with a unigram language model. inverted[i] stores
        // the set of sentence index where the sentencepieces[i] appears.
        let chunk_size = std::cmp::max(sentences.len() / current_num_threads(), 1);
        let indexed_sentences: Vec<(usize, &Sentence)> = sentences.iter().enumerate().collect();
        let collected: (f64, Vec<f64>, Vec<Vec<usize>>) = indexed_sentences
            .maybe_par_chunks(chunk_size)
            .map(|enumerated_sentence_count_chunk| {
                let mut vsum = 0.0;
                let mut freq: Vec<f64> = vec![0.0; pieces.len()];
                let mut inverted: Vec<Vec<usize>> = vec![Vec::new(); pieces.len()];

                for (i, (sentence, count)) in enumerated_sentence_count_chunk {
                    let mut lattice = Lattice::from(sentence, bos_id, eos_id);
                    model.populate_nodes(&mut lattice);
                    vsum += *count as f64;
                    for node_ref in lattice.viterbi() {
                        let id = node_ref.borrow().id;
                        freq[id] += *count as f64;
                        inverted[id].push(*i);
                    }
                }
                (vsum, freq, inverted)
            })
            .reduce(
                || (0.0, vec![0.0; pieces.len()], vec![Vec::new(); pieces.len()]),
                |(vsum, freq, inverted), (lvsum, lfreq, linverted)| {
                    (
                        vsum + lvsum,
                        freq.iter()
                            .zip(lfreq)
                            .map(|(global_el, local_el)| global_el + local_el)
                            .collect(),
                        inverted
                            .iter()
                            .zip(linverted)
                            .map(|(global_el, local_el)| [&global_el[..], &local_el[..]].concat())
                            .collect(),
                    )
                },
            );

        let (vsum, freq, inverted) = collected;

        let sum: f64 = freq.iter().sum();
        let logsum = sum.ln();
        let mut candidates: Vec<(usize, f64)> = vec![];
        let mut new_pieces: Vec<SentencePiece> = Vec::with_capacity(self.vocab_size as usize);
        new_pieces.push(pieces[0].clone());

        // Finally, computes how likely the LM likelihood is reduced if
        // the sentencepiece[i] is removed from the vocabulary.
        // Since the exact computation of loss is difficult, we compute the
        // loss approximately by assuming that all sentencepiece[i] in the sentences
        // are replaced with alternatives[i] when sentencepiece[i] is removed.
        for (id, (token, score)) in pieces.iter().enumerate() {
            if id == 0 {
                continue;
            }
            if freq[id] == 0.0 && !always_keep[id] {
                // not found in Viterbi path. Can remove this entry safely.
                continue;
            } else if alternatives[id].is_empty() {
                // no alternatives. Keeps this entry.
                new_pieces.push((token.to_string(), *score));
            } else {
                let mut f = 0.0; // the frequency of pieces[i];

                for n in &inverted[id] {
                    let score = sentences[*n].1 as f64;
                    f += score;
                }
                // TODO: Temporary hack to avoid Nans.
                if f == 0.0 || f.is_nan() {
                    // new_pieces.push((token.to_string(), *score));
                    continue;
                }
                f /= vsum; // normalizes by all sentence frequency.
                let logprob_sp = freq[id].ln() - logsum;

                // After removing the sentencepiece[i], its frequency freq[i] is
                // re-assigned to alternatives.
                // new_sum = current_sum - freq[i] + freq[i] * alternatives.size()
                //         = current_sum + freq[i] (alternatives - 1)

                let logsum_alt = (sum + freq[id] * (alternatives.len() - 1) as f64).ln();

                // The frequencies of alternatives are increased by freq[i].
                let mut logprob_alt = 0.0;
                for n in &alternatives[id] {
                    logprob_alt += (freq[*n] + freq[id]).ln() - logsum_alt;
                }

                // loss: the diff of likelihood after removing the sentencepieces[i].
                let loss = f * (logprob_sp - logprob_alt);
                if loss.is_nan() {
                    panic!("");
                }

                candidates.push((id, loss));
            }
        }
        let desired_vocab_size: usize = (self.vocab_size as usize * 11) / 10; // * 1.1
        let pruned_size: usize = ((pieces.len() as f64) * self.shrinking_factor) as usize;
        let pruned_size = desired_vocab_size.max(pruned_size);

        candidates.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        for (id, _score) in candidates {
            if new_pieces.len() == pruned_size {
                break;
            }
            new_pieces.push(pieces[id].clone());
        }

        new_pieces
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

    fn run_e_step(&self, model: &Unigram, sentences: &[Sentence]) -> (f64, u32, Vec<f64>) {
        let all_sentence_freq: u32 = sentences.iter().map(|(_a, b)| *b).sum();

        let chunk_size = std::cmp::max(sentences.len() / current_num_threads(), 1);
        let collected: (f64, u32, Vec<f64>) = sentences
            .maybe_par_chunks(chunk_size)
            .map(|sentences_chunk| {
                let mut expected: Vec<f64> = vec![0.0; model.len()];
                let mut objs: f64 = 0.0;
                let mut ntokens: u32 = 0;

                for (string, freq) in sentences_chunk {
                    let mut lattice = Lattice::from(string, model.bos_id, model.eos_id);
                    model.populate_nodes(&mut lattice);

                    let z: f64 = lattice.populate_marginal(*freq as f64, &mut expected);
                    if z.is_nan() {
                        panic!("likelihood is NAN. Input sentence may be too long.");
                    }
                    ntokens += lattice.viterbi().len() as u32;
                    objs -= z / (all_sentence_freq as f64);
                }
                (objs, ntokens, expected)
            })
            .reduce(
                || (0.0, 0, vec![0.0; model.len()]),
                |(objs, ntokens, expected), (lobjs, lntokens, lexpected)| {
                    (
                        objs + lobjs,
                        ntokens + lntokens,
                        expected
                            .iter()
                            .zip(lexpected)
                            .map(|(global_el, local_el)| global_el + local_el)
                            .collect(),
                    )
                },
            );

        collected
    }
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
        // // Here we do not use the original EM, but use the
        // // Bayesianified/DPified EM algorithm.
        // // https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
        // // This modification will act as a sparse prior.
        let logsum = digamma(sum);
        let new_pieces: Vec<_> = new_pieces
            .into_iter()
            .map(|(s, c)| (s, digamma(c) - logsum))
            .collect();
        new_pieces
    }
    pub fn do_train(
        &self,
        sentences: Vec<Sentence>,
        model: &mut Unigram,
    ) -> Result<Vec<AddedToken>> {
        let progress = self.setup_progress();
        //
        // 1. Compute frequent substrings
        // TODO Should be able to upgrade to u64 when needed
        self.update_progress(&progress, sentences.len(), "Suffix array seeds");
        let mut pieces: Vec<SentencePiece> =
            Vec::with_capacity(self.vocab_size.try_into().unwrap());

        // We use a UNK token when training, whatever the `self.unk_token`
        pieces.push(("<UNK>".into(), f64::NAN));
        pieces.extend(seed::make_seed_pieces(
            &sentences,
            self.seed_init,
            self.seed_size.unwrap_or(seed::DEFAULT_SEED_SIZE),
            self.max_piece_length,
            &self.initial_alphabet,
            &progress,
        )?);
        self.finalize_progress(&progress, sentences.len());

        // Useful to check compatibility with spm.
        debug!(
            "Using {} pieces on {} sentences for EM training",
            pieces.len(),
            sentences.len()
        );

        let desired_vocab_size: usize = (self.vocab_size as usize * 11) / 10; // * 1.1

        // 2. Run E-M Loops to fine grain the pieces.
        // We will shrink the vocab by shrinking_factor every loop on average
        // Some other pieces are dropped if logprob is too small
        // V = N * (f)**k
        // k = log(V / N) / log(f)
        let expected_loops = (((desired_vocab_size as f64).ln() - (pieces.len() as f64).ln())
            / self.shrinking_factor.ln()) as usize
            + 1;
        let expected_updates = expected_loops * self.n_sub_iterations as usize;
        self.update_progress(&progress, expected_updates, "EM training");
        let required_chars = seed::required_chars(&sentences, &self.initial_alphabet);
        if required_chars.len() as u32 > self.vocab_size {
            return Err(Box::new(UnigramTrainerError::VocabularyTooSmall));
        }
        let mut new_model = Unigram::from(pieces.clone(), Some(0), false)?;
        loop {
            // Sub-EM iteration.
            for _iter in 0..self.n_sub_iterations {
                // Executes E step
                let (_objective, _num_tokens, expected) = self.run_e_step(&new_model, &sentences);

                // Executes M step.
                pieces = self.run_m_step(&pieces, &expected);
                new_model = Unigram::from(pieces.clone(), Some(0), false)?;

                // Useful comment for checking compatibility with spm
                debug!(
                    "Em iter={} size={} obj={} num_tokens={} num_tokens/piece={}",
                    _iter,
                    new_model.len(),
                    _objective,
                    _num_tokens,
                    _num_tokens as f64 / model.len() as f64
                );
                if let Some(p) = &progress {
                    p.inc(1);
                }
            } // end of Sub EM iteration

            // Stops the iteration when the size of sentences reaches to the
            // desired symbol size.
            if pieces.len() <= desired_vocab_size {
                break;
            }

            // Prunes pieces.
            pieces = self.prune_sentence_pieces(&new_model, &pieces, &sentences);
            new_model = Unigram::from(pieces.clone(), Some(0), false)?;
        }
        self.finalize_progress(&progress, expected_updates);

        // Finally, adjusts the size of sentencepices to be |vocab_size|.
        *model = seed::finalize(
            self.vocab_size,
            &self.special_tokens,
            &self.unk_token,
            &new_model,
            required_chars,
        )?;

        Ok(self.special_tokens.clone())
    }
}

impl Trainer for UnigramTrainer {
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

    #[test]
    fn test_unk_token() {
        // 1. Should add `unk_token` as first special token
        let trainer = UnigramTrainerBuilder::default()
            .show_progress(false)
            .special_tokens(vec![
                AddedToken::from("[SEP]", true),
                AddedToken::from("[CLS]", true),
            ])
            .unk_token(Some("[UNK]".into()))
            .build()
            .unwrap();

        let mut unigram = Unigram::default();
        trainer
            .do_train(vec![("The".into(), 12), ("are".into(), 11)], &mut unigram)
            .unwrap();

        let mut pieces = unigram.iter();
        assert_eq!(pieces.next(), Some(&("[UNK]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[SEP]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[CLS]".into(), 0.0)));

        // 2. Let it where it is
        let trainer = UnigramTrainerBuilder::default()
            .show_progress(false)
            .special_tokens(vec![
                AddedToken::from("[SEP]", true),
                AddedToken::from("[CLS]", true),
                AddedToken::from("[UNK]", true),
            ])
            .unk_token(Some("[UNK]".into()))
            .build()
            .unwrap();

        let mut unigram = Unigram::default();
        trainer
            .do_train(vec![("The".into(), 12), ("are".into(), 11)], &mut unigram)
            .unwrap();

        let mut pieces = unigram.iter();
        assert_eq!(pieces.next(), Some(&("[SEP]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[CLS]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[UNK]".into(), 0.0)));

        // 3. Don't put it there if not needed
        let trainer = UnigramTrainerBuilder::default()
            .show_progress(false)
            .build()
            .unwrap();

        let mut unigram = Unigram::default();
        trainer
            .do_train(vec![("The".into(), 12), ("are".into(), 11)], &mut unigram)
            .unwrap();

        let mut pieces = unigram.iter();
        assert_eq!(pieces.next().unwrap().0, "e".to_string());
    }

    #[test]
    fn test_special_tokens() {
        let trainer = UnigramTrainerBuilder::default()
            .show_progress(false)
            .special_tokens(vec![
                AddedToken::from("[SEP]", true),
                AddedToken::from("[CLS]", true),
            ])
            .build()
            .unwrap();

        let mut unigram = Unigram::default();
        trainer
            .do_train(vec![("The".into(), 12), ("are".into(), 11)], &mut unigram)
            .unwrap();

        let mut pieces = unigram.iter();
        assert_eq!(pieces.next(), Some(&("[SEP]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[CLS]".into(), 0.0)));
    }

    #[test]
    fn test_seed_init_default_is_substrings() {
        let trainer = UnigramTrainer::default();
        assert_eq!(trainer.seed_init, SeedInit::Substrings);
    }

    #[test]
    fn test_seed_size_default_is_none() {
        assert_eq!(UnigramTrainer::default().seed_size, None);
        let trainer = UnigramTrainer::builder().seed_size(30).build().unwrap();
        assert_eq!(trainer.seed_size, Some(30));
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn test_unigram_trains_with_bpe_seed() {
        let trainer = UnigramTrainer::builder()
            .show_progress(false)
            .vocab_size(300)
            .seed_size(30)
            .seed_init(SeedInit::Bpe)
            .build()
            .unwrap();

        let mut unigram = Unigram::default();
        trainer
            .do_train(
                vec![("hello".into(), 10), ("world".into(), 8)],
                &mut unigram,
            )
            .unwrap();

        assert!(unigram.get_vocab_size() <= 300);
        for c in ["h", "e", "l", "o", "w", "r", "d"] {
            assert!(unigram.token_to_id(c).is_some(), "missing char {}", c);
        }
        // Cross-combination: a UnigramTrainer model encodes under min_tokens.
        unigram.set_objective(tk_encode::models::unigram::Objective::MinTokens);
        assert!(!unigram.encode("hello").unwrap().is_empty());
    }

    #[test]
    fn test_byte_floor_in_budget() {
        let corpus = || -> Vec<Sentence> {
            vec![("hello".into(), 10), ("日本語".into(), 8), ("🎉".into(), 3)]
        };
        let trainer = UnigramTrainer::builder()
            .show_progress(false)
            .vocab_size(300)
            .seed_size(40)
            .build()
            .unwrap();
        let mut unigram = Unigram::default();
        trainer.do_train(corpus(), &mut unigram).unwrap();
        let spellings: Vec<&str> = unigram.iter().map(|(t, _)| t.as_str()).collect();
        // The floor counts toward vocab_size: never overshoot it.
        assert!(spellings.len() <= 300, "overshot: {}", spellings.len());
        // 日 E697A5 -> E697; 🎉 F09F8E89 -> F09F, F09F8E.
        for piece in [
            "<0x00>",
            "<0x0A>",
            "<0x71>",
            "<0xF4>",
            "<0xE697>",
            "<0xF09F>",
            "<0xF09F8E>",
        ] {
            assert!(spellings.contains(&piece), "missing {piece}");
        }
        // Seen ASCII keeps its real piece: `h` tiles `<0x68>` already.
        for missing in ["<0x68>", "<0xC0>", "<0xC1>", "<0xF5>", "<0xFF>"] {
            assert!(!spellings.contains(&missing), "{missing} must be absent");
        }
    }
}
