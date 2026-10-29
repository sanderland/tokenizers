//! MinGram ("Minimalist Unigram") trainer: BPE-seeded hard-EM training that
//! outputs a standard `Unigram` model with `objective = min_tokens`.
//!
//! Seed -> outer EM/prune loop (one shrink step by default, matching
//! script_tok's default schedule) -> shared finalize. No forward-backward
//! pass. Pruning drops the least-used pieces by default, with an optional
//! minimum-increase port of script_tok's `_mi_prune.py`.

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

/// What a prune round drops: the least-used pieces (default), or the pieces
/// whose removal adds the fewest extra corpus tokens (minimum-increase,
/// Schmidt et al. 2024, ported from `script_tok/.../_mi_prune.py`, where it
/// is shared with PathPiece).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PruneCriterion {
    #[default]
    UsageCount,
    MinimumIncrease,
}

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
    seed_size: Option<usize>,
    pub seed_init: SeedInit,
    pub prune_criterion: PruneCriterion,
    pub pruning_shrinking_factor: f64,
    pub skip_substring_in_batch: bool,
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
    prune_criterion: Option<PruneCriterion>,
    pruning_shrinking_factor: Option<f64>,
    skip_substring_in_batch: Option<bool>,
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

    /// Prune the least-used pieces (default) or minimum-increase. MI pairs
    /// best with iterative pruning (`pruning_shrinking_factor` above 0).
    pub fn prune_criterion(&mut self, prune_criterion: PruneCriterion) -> &mut Self {
        self.prune_criterion = Some(prune_criterion);
        self
    }

    /// Shrink step per outer round: 0 prunes to the final size in one step
    /// (matching script_tok's default schedule); above 0 the prune is
    /// gradual and each round re-runs the EM sub-iterations.
    pub fn pruning_shrinking_factor(&mut self, pruning_shrinking_factor: f64) -> &mut Self {
        self.pruning_shrinking_factor = Some(pruning_shrinking_factor);
        self
    }

    /// In an MI drop batch, skip a candidate contained in an already-dropped
    /// token. MI pruning only; ignored under usage-count pruning.
    pub fn skip_substring_in_batch(&mut self, skip_substring_in_batch: bool) -> &mut Self {
        self.skip_substring_in_batch = Some(skip_substring_in_batch);
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
            seed_size: self.seed_size,
            seed_init: self.seed_init.unwrap_or(SeedInit::Bpe),
            prune_criterion: self.prune_criterion.unwrap_or_default(),
            pruning_shrinking_factor: self.pruning_shrinking_factor.unwrap_or(0.0),
            skip_substring_in_batch: self.skip_substring_in_batch.unwrap_or(true),
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

/// Char-trie over training pieces for MI match enumeration. The Unigram
/// byte-trie is crate-private and byte-indexed; MI needs char-indexed
/// matches, so a small dedicated trie is simpler than reaching in.
struct CharTrie {
    children: Vec<AHashMap<char, usize>>,
    terminal: Vec<Option<usize>>,
}

impl CharTrie {
    fn build(pieces: &[SentencePiece]) -> Self {
        let mut trie = Self {
            children: vec![AHashMap::new()],
            terminal: vec![None],
        };
        for (idx, (piece, _)) in pieces.iter().enumerate().skip(1) {
            let mut node = 0;
            for c in piece.chars() {
                if let Some(&next) = trie.children[node].get(&c) {
                    node = next;
                } else {
                    let next = trie.children.len();
                    trie.children.push(AHashMap::new());
                    trie.terminal.push(None);
                    trie.children[node].insert(c, next);
                    node = next;
                }
            }
            trie.terminal[node] = Some(idx);
        }
        trie
    }

    /// `(char_len, piece_idx)` of every piece matching a prefix of `chars`,
    /// shortest first.
    fn matches(&self, chars: &[char]) -> Vec<(usize, usize)> {
        let mut out = vec![];
        let mut node = 0;
        for (i, c) in chars.iter().enumerate() {
            match self.children[node].get(c) {
                Some(&next) => {
                    node = next;
                    if let Some(idx) = self.terminal[node] {
                        out.push((i + 1, idx));
                    }
                }
                None => break,
            }
        }
        out
    }
}

/// Minimum-increase lower bound per piece: how many extra corpus tokens
/// dropping the piece would add. Ported from `script_tok/.../_mi_prune.py`
/// (Schmidt et al. 2024 section 3.2): for each occurrence in the model's own
/// segmentation, the cheaper of re-tiling across one internal break or
/// replacing the piece with a strictly larger match covering it. Indexed by
/// piece; the UNK slot and the required singles keep 0.0.
fn compute_mi_table(pieces: &[SentencePiece], sentences: &[Sentence], model: &Unigram) -> Vec<f64> {
    let trie = CharTrie::build(pieces);
    let max_width = pieces[1..]
        .iter()
        .map(|(piece, _)| piece.chars().count())
        .max()
        .unwrap_or(0);
    let mut mi = vec![0.0; pieces.len()];
    for (word, freq) in sentences {
        let chars: Vec<char> = word.chars().collect();
        let n = chars.len();
        if n == 0 {
            continue;
        }
        let unreachable = n + 1;
        // Forward min-path lengths plus every match ending at each position.
        let mut pl = vec![unreachable; n + 1];
        pl[0] = 0;
        let mut ending_at: Vec<Vec<(usize, usize)>> = vec![vec![]; n + 1];
        for s in 0..n {
            let base = pl[s];
            for (w, idx) in trie.matches(&chars[s..]) {
                let e = s + w;
                ending_at[e].push((w, idx));
                if base < unreachable && base + 1 < pl[e] {
                    pl[e] = base + 1;
                }
            }
        }
        let k_d = pl[n];
        if k_d >= unreachable {
            continue;
        }
        // Backward min-path lengths.
        let mut bpl = vec![unreachable; n + 1];
        bpl[n] = 0;
        for e in (0..n).rev() {
            let mut best = unreachable;
            for (w, _) in trie.matches(&chars[e..]) {
                let cand = bpl[e + w];
                if cand < unreachable && cand + 1 < best {
                    best = cand + 1;
                }
            }
            bpl[e] = best;
        }
        // The model's own segmentation into char ranges; words it cannot
        // cover without UNK are skipped (impossible while the M-step keeps
        // every observed single).
        let mut lattice = Lattice::from(word, model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);
        let path = lattice.viterbi_min_tokens();
        let mut ranges: Vec<(usize, usize, usize)> = vec![];
        let mut pos = 0;
        let mut usable = true;
        for node_ref in &path {
            let id = node_ref.borrow().id;
            if id == 0 || id >= pieces.len() {
                usable = false;
                break;
            }
            let w = pieces[id].0.chars().count();
            ranges.push((pos, pos + w, id));
            pos += w;
        }
        if !usable || pos != n {
            continue;
        }
        for (s, e, idx) in ranges {
            if e - s == 1 {
                continue; // required singles are never pruned
            }
            let mut best_break = unreachable;
            for j in s + 1..e {
                best_break = best_break.min(pl[j] + bpl[j]);
            }
            let mi_b = best_break - k_d;
            let mut mi_s = unreachable;
            for ep in e..(s + max_width).min(n) + 1 {
                for &(w_prime, _) in &ending_at[ep] {
                    let sp = ep - w_prime;
                    if sp > s || (sp == s && ep == e) {
                        continue;
                    }
                    mi_s = mi_s.min(pl[sp] + bpl[ep] + 1);
                }
            }
            mi_s -= k_d;
            let local_mi = mi_b.min(mi_s);
            let local = if local_mi >= unreachable {
                f64::INFINITY
            } else {
                local_mi as f64
            };
            mi[idx] += *freq as f64 * local;
        }
    }
    mi
}

/// Drop batch used by MI pruning: the first `k` candidates in MI order,
/// skipping a candidate already contained in a dropped token when
/// `skip_substring` holds (script_tok `_mi_prune.select_drop_batch`).
fn select_drop_batch(
    ordered_ids: &[usize],
    k: usize,
    pieces: &[SentencePiece],
    skip_substring: bool,
) -> AHashSet<usize> {
    const MAX_SUB_LEN: usize = 16;
    if !skip_substring {
        return ordered_ids.iter().take(k).copied().collect();
    }
    let mut dropped = AHashSet::new();
    let mut sub_seqs: AHashSet<Vec<char>> = AHashSet::new();
    for &idx in ordered_ids {
        if dropped.len() >= k {
            break;
        }
        let seq: Vec<char> = pieces[idx].0.chars().collect();
        if seq.len() <= MAX_SUB_LEN && sub_seqs.contains(&seq) {
            continue;
        }
        dropped.insert(idx);
        for i in 0..seq.len() {
            for j in i + 1..=(i + MAX_SUB_LEN).min(seq.len()) {
                sub_seqs.insert(seq[i..j].to_vec());
            }
        }
    }
    dropped
}

/// Drop down to `target_size` by minimum increase, lowest first, breaking
/// ties by current usage score (script_tok `_mi_prune.mi_prune`). UNK and
/// the observed singles are kept unconditionally.
fn mi_prune(
    pieces: &[SentencePiece],
    sentences: &[Sentence],
    model: &Unigram,
    target_size: usize,
    skip_substring: bool,
) -> Vec<SentencePiece> {
    let n_required = 1 + pieces[1..]
        .iter()
        .filter(|(piece, _)| piece.chars().count() == 1)
        .count();
    let n_remove = (pieces.len().saturating_sub(target_size)).min(pieces.len() - n_required);
    if n_remove == 0 {
        return pieces.to_vec();
    }
    let mi = compute_mi_table(pieces, sentences, model);
    let mut candidates: Vec<usize> = (1..pieces.len())
        .filter(|&i| pieces[i].0.chars().count() != 1)
        .collect();
    candidates.sort_by(|&a, &b| {
        mi[a]
            .partial_cmp(&mi[b])
            .unwrap()
            .then(pieces[a].1.partial_cmp(&pieces[b].1).unwrap())
    });
    let drop = select_drop_batch(&candidates, n_remove, pieces, skip_substring);
    pieces
        .iter()
        .enumerate()
        .filter(|(i, _)| !drop.contains(i))
        .map(|(_, piece)| piece.clone())
        .collect()
}

impl MinGramTrainer {
    /// Outer-loop cap, mirroring script_tok's `MAX_ITERATIONS`.
    const MAX_ITERATIONS: usize = 100;

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
                        // Fallback nodes have no vocab slot; like UNK mass
                        // (dropped from the total below) they count nowhere.
                        if id < expected.len() {
                            expected[id] += *freq as f64;
                        }
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
    /// plain MLE log-probs (no Bayesian prior). Single chars are the
    /// alphabet and never drop (floored at 0.5 for scoring).
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
            if *freq < expected_frequency_threshold && piece.chars().count() != 1 {
                continue;
            }
            let freq = (*freq).max(expected_frequency_threshold);
            new_pieces.push((piece.clone(), freq));
            sum += freq;
        }
        let logsum = sum.ln();
        new_pieces
            .into_iter()
            .map(|(s, c)| (s, c.ln() - logsum))
            .collect()
    }

    /// Prune to `target_size` under the configured criterion (MI pruning
    /// needs the current model for the segmentation its bounds are built on).
    fn prune_to_target(
        &self,
        pieces: &[SentencePiece],
        sentences: &[Sentence],
        model: &Unigram,
        target_size: usize,
    ) -> Vec<SentencePiece> {
        match self.prune_criterion {
            PruneCriterion::UsageCount => flat_prune(pieces, target_size),
            PruneCriterion::MinimumIncrease => mi_prune(
                pieces,
                sentences,
                model,
                target_size,
                self.skip_substring_in_batch,
            ),
        }
    }

    /// Outer shrink step (script_tok `prune_tokens`): with a zero factor the
    /// target is the final size, so one outer round prunes everything and a
    /// second re-estimates on it; above zero the prune shrinks gradually.
    fn shrink_target(&self, pieces: &[SentencePiece], final_size: usize) -> usize {
        let n_singles = pieces[1..]
            .iter()
            .filter(|(piece, _)| piece.chars().count() == 1)
            .count();
        let non_atomic = pieces.len() - 1 - n_singles;
        let shrink_n = (non_atomic as f64 * (1.0 - self.pruning_shrinking_factor)) as usize;
        final_size.max(non_atomic.saturating_sub(shrink_n))
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
            self.seed_size.unwrap_or(seed::DEFAULT_SEED_SIZE),
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

        // 2. Outer EM/prune loop (script_tok schedule): EM sub-iterations,
        // break at size, shrink-step prune. A zero shrinking factor prunes
        // to the final size in the first round and re-estimates once more;
        // above zero the bar pins early and further rounds run past it.
        let final_size = self.vocab_size as usize;
        let expected_updates = self.n_sub_iterations as usize * 2;
        self.update_progress(&progress, expected_updates, "Hard EM training");
        let mut new_model = Unigram::from(pieces.clone(), Some(0), false)?;
        for outer in 0..Self::MAX_ITERATIONS {
            for _iter in 0..self.n_sub_iterations {
                let (num_tokens, expected) = self.run_e_step(&new_model, &sentences);
                pieces = self.run_m_step(&pieces, &expected);
                new_model = Unigram::from(pieces.clone(), Some(0), false)?;

                debug!(
                    "Hard EM outer={} iter={} size={} num_tokens={}",
                    outer,
                    _iter,
                    new_model.len(),
                    num_tokens,
                );
                if let Some(p) = &progress {
                    p.inc(1);
                }
            }
            if pieces.len() <= final_size {
                break;
            }
            pieces = self.prune_to_target(
                &pieces,
                &sentences,
                &new_model,
                self.shrink_target(&pieces, final_size),
            );
            new_model = Unigram::from(pieces.clone(), Some(0), false)?;
        }
        self.finalize_progress(&progress, expected_updates);

        // 3. Final exact prune (criterion-dispatched) down to `vocab_size`.
        pieces = self.prune_to_target(&pieces, &sentences, &new_model, final_size);
        new_model = Unigram::from(pieces.clone(), Some(0), false)?;

        // 4. Adjust the size of sentencepieces to be |vocab_size|.
        *model = seed::finalize(
            self.vocab_size,
            &self.special_tokens,
            &self.unk_token,
            &new_model,
            required_chars,
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
    fn test_e_step_skips_fallback_without_panicking() {
        // A no-unk model on an unseen char: the fallback node has no vocab
        // slot, so it counts nowhere instead of indexing out of bounds.
        let model = Unigram::from(vec![("a".into(), 0.0)], None, false).unwrap();
        let trainer = MinGramTrainer::builder().build().unwrap();
        let (ntokens, expected) = trainer.run_e_step(&model, &[("z".to_string(), 3)]);
        assert_eq!(ntokens, 1);
        assert_eq!(expected, vec![0.0]);
    }

    #[test]
    fn test_m_step_drops_zero_counts_keeps_singles() {
        let trainer = MinGramTrainer::builder().build().unwrap();
        let pieces = vec![
            ("<UNK>".to_string(), f64::NAN),
            ("a".to_string(), -1.0),
            ("b".to_string(), -1.0),
            ("ab".to_string(), -1.0),
        ];
        let new_pieces = trainer.run_m_step(&pieces, &[0.0, 4.0, 0.0, 2.0]);

        // "b" has zero counts but is a single char: kept, floored at 0.5.
        assert_eq!(new_pieces.len(), 4);
        assert_eq!(new_pieces[0].0, "<UNK>");
        assert!(new_pieces[0].1.is_nan());
        assert_eq!(new_pieces[1].0, "a");
        assert_approx_eq!(new_pieces[1].1, (4.0_f64 / 6.5).ln(), 1e-9);
        assert_eq!(new_pieces[2].0, "b");
        assert_approx_eq!(new_pieces[2].1, (0.5_f64 / 6.5).ln(), 1e-9);
        assert_eq!(new_pieces[3].0, "ab");
        assert_approx_eq!(new_pieces[3].1, (2.0_f64 / 6.5).ln(), 1e-9);
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

    #[test]
    fn test_seed_size_default_is_none() {
        assert_eq!(MinGramTrainer::default().seed_size, None);
        let trainer = MinGramTrainer::builder().seed_size(30).build().unwrap();
        assert_eq!(trainer.seed_size, Some(30));
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn test_mingram_trains_with_bpe_seed() {
        let trainer = MinGramTrainer::builder()
            .show_progress(false)
            .vocab_size(300)
            .seed_size(30)
            .build()
            .unwrap();
        assert_eq!(trainer.seed_init, SeedInit::Bpe);

        let mut model = Unigram::default();
        trainer.do_train(tiny_corpus(), &mut model).unwrap();

        assert_eq!(model.objective(), Objective::MinTokens);
        assert!(model.get_vocab_size() <= 300);
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
            .vocab_size(300)
            .seed_init(SeedInit::Substrings)
            .build()
            .unwrap();

        let mut model = Unigram::default();
        trainer.do_train(tiny_corpus(), &mut model).unwrap();

        assert_eq!(model.objective(), Objective::MinTokens);
        assert!(model.get_vocab_size() <= 300);
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
            .vocab_size(300)
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

    #[test]
    fn test_mi_prune_drops_low_mi_first() {
        // "ab" x10 + "abcdef" x3: usage-count would drop abcdef (3 < 10),
        // but removing it costs 12 extra tokens vs 10 for "ab", so minimum
        // increase drops "ab" (MI 10 < 12; the [ab] shortcut in "abcdef"
        // tiles it in 5, hence 12 rather than 15).
        let pieces: Vec<SentencePiece> = ["<UNK>", "a", "b", "c", "d", "e", "f", "ab", "abcdef"]
            .iter()
            .map(|piece| (piece.to_string(), 0.0))
            .collect();
        let mut scored = pieces.clone();
        scored[7].1 = 0.0; // ab: frequent...
        scored[8].1 = -10.0; // abcdef: rare (usage-count would drop it)
        let sentences = vec![("ab".to_string(), 10), ("abcdef".to_string(), 3)];
        let model = Unigram::from(scored.clone(), Some(0), false).unwrap();
        // Table values verified against script_tok's `compute_mi_table` on
        // the same pieces, counts, and segmentation (ab 10.0, abcdef 12.0).
        let mi = compute_mi_table(&scored, &sentences, &model);
        assert_eq!(mi, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 12.0]);
        let kept = mi_prune(&scored, &sentences, &model, 8, false);
        let spellings: Vec<&str> = kept.iter().map(|(piece, _)| piece.as_str()).collect();
        assert!(!spellings.contains(&"ab"));
        assert!(spellings.contains(&"abcdef"));
        // Same setup under usage-count pruning drops the rare piece instead.
        let kept_usage = flat_prune(&scored, 8);
        let spellings: Vec<&str> = kept_usage.iter().map(|(piece, _)| piece.as_str()).collect();
        assert!(spellings.contains(&"ab"));
        assert!(!spellings.contains(&"abcdef"));
    }

    #[test]
    fn test_select_drop_batch_skip_substring() {
        let pieces: Vec<SentencePiece> = ["<UNK>", "a", "b", "c", "ab", "abc"]
            .iter()
            .map(|piece| (piece.to_string(), 0.0))
            .collect();
        let ordered = [5, 4]; // abc first; ab is contained in it
        assert_eq!(
            select_drop_batch(&ordered, 2, &pieces, false),
            AHashSet::from([4, 5])
        );
        assert_eq!(
            select_drop_batch(&ordered, 2, &pieces, true),
            AHashSet::from([5])
        );
    }

    #[test]
    fn test_mingram_mi_prune_trains() {
        // MI pruning with gradual shrinking exercises the multi-outer loop;
        // the default single-step schedule is covered by the tests below.
        let corpus = || -> Vec<Sentence> {
            vec![
                ("hello".into(), 10),
                ("world".into(), 8),
                ("hello world".into(), 3),
            ]
        };
        let train = || {
            let trainer = MinGramTrainer::builder()
                .show_progress(false)
                .vocab_size(300)
                .seed_size(30)
                .seed_init(SeedInit::Substrings)
                .prune_criterion(PruneCriterion::MinimumIncrease)
                .pruning_shrinking_factor(0.5)
                .skip_substring_in_batch(true)
                .build()
                .unwrap();
            let mut model = Unigram::default();
            trainer.do_train(corpus(), &mut model).unwrap();
            model.iter().cloned().collect::<Vec<_>>()
        };
        let once = train();
        assert!(once.len() <= 300);
        assert_eq!(train(), once);
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn test_mingram_is_deterministic() {
        let train = || {
            let trainer = MinGramTrainer::builder()
                .show_progress(false)
                .vocab_size(300)
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
    fn test_byte_floor_in_budget() {
        let corpus = || -> Vec<Sentence> {
            vec![("hello".into(), 10), ("日本語".into(), 8), ("🎉".into(), 3)]
        };
        let train = || {
            let trainer = MinGramTrainer::builder()
                .show_progress(false)
                .vocab_size(300)
                .seed_size(40)
                .seed_init(SeedInit::Substrings)
                .build()
                .unwrap();
            let mut model = Unigram::default();
            trainer.do_train(corpus(), &mut model).unwrap();
            model.iter().cloned().collect::<Vec<_>>()
        };
        let once = train();
        // The floor counts toward vocab_size: never overshoot it.
        assert!(once.len() <= 300, "overshot: {}", once.len());
        let spellings: Vec<&str> = once.iter().map(|(t, _)| t.as_str()).collect();
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
        assert_eq!(train(), once);
    }
}
