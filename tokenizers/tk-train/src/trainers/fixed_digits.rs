//! Shared fixed numeric vocabulary reservation; numeric chunks never enter EM.
use super::seed::{self, AlphabetMode, Sentence};
use ahash::AHashSet;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tk_encode::{
    Result,
    models::unigram::Unigram,
    pipeline::{PreTokenizer, PreTokenizerScratch},
    pre_tokenizers::fixed_digits::FixedDigits,
    vocab::bucket_added_vocabulary::AddedToken,
};

pub(crate) mod serde_policy {
    use super::*;
    pub fn serialize<S: Serializer>(
        policy: &Option<FixedDigits>,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        policy
            .map(|p| (p.group_size, p.boundary_markers))
            .serialize(serializer)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Option<FixedDigits>, D::Error> {
        let raw = Option::<(usize, bool)>::deserialize(deserializer)?;
        raw.map(|(group_size, boundary_markers)| {
            let p = FixedDigits {
                group_size,
                boundary_markers,
            };
            p.validate().map_err(serde::de::Error::custom)?;
            Ok(p)
        })
        .transpose()
    }
}
pub(crate) struct Prepared {
    pub sentences: Vec<Sentence>,
    pub alphabet: AHashSet<char>,
    pub extras: Vec<String>,
    pub vocab_size: u32,
    pub inventory: AHashSet<String>,
}
pub(crate) fn prepare(
    sentences: Vec<Sentence>,
    initial: &AHashSet<char>,
    mode: AlphabetMode,
    policy: FixedDigits,
    vocab_size: u32,
    specials: &[AddedToken],
) -> Result<Prepared> {
    policy.validate()?;
    if mode == AlphabetMode::ByteLevel && policy.boundary_markers {
        return Err("marked fixed digits require the UTF-8 or legacy alphabet".into());
    }
    if !sentences.iter().any(|(s, n)| !s.is_empty() && *n > 0) {
        return Err("fixed-digit training requires nonempty text with positive counts".into());
    }
    let inventory: AHashSet<String> = policy.inventory().into_iter().collect();
    if specials.iter().any(|s| inventory.contains(&s.content)) {
        return Err("special token collides with fixed digit inventory".into());
    }
    let mut filtered = Vec::new();
    for (text, count) in sentences {
        if count == 0 {
            continue;
        }
        let mut spans = Vec::new();
        policy.pre_tokenize(&text, &mut PreTokenizerScratch::default(), &mut spans)?;
        for span in spans {
            let chunk = &text[span.range()];
            if !inventory.contains(chunk) && !chunk.is_empty() {
                filtered.push((chunk.to_owned(), count));
            }
        }
    }
    let mut alphabet = initial.clone();
    alphabet.extend('0'..='9');
    let required = seed::required_chars(&filtered, &alphabet, mode);
    let mut extras: Vec<_> = inventory
        .iter()
        .filter(|s| !required.contains(*s))
        .cloned()
        .collect();
    extras.sort();
    let vocab_size = vocab_size
        .checked_sub(extras.len() as u32)
        .ok_or("vocabulary cannot fit fixed digit inventory")?;
    Ok(Prepared {
        sentences: filtered,
        alphabet,
        extras,
        vocab_size,
        inventory,
    })
}
pub(crate) fn empty_model(
    prepared: &Prepared,
    mode: AlphabetMode,
    specials: &[AddedToken],
    unk: &Option<String>,
) -> Result<Unigram> {
    let required = seed::required_chars(&[], &prepared.alphabet, mode);
    let pieces = required.iter().map(|s| (s.clone(), -100.)).collect();
    seed::finalize(
        prepared.vocab_size,
        specials,
        unk,
        &Unigram::from(pieces, None, false)?,
        required,
        mode,
    )
}
pub(crate) fn finish(model: &mut Unigram, prepared: Prepared) -> Result<()> {
    let objective = model.objective();
    let mut pieces = model.vocab().to_vec();
    for (s, score) in &mut pieces {
        if prepared.inventory.contains(s) {
            *score = 0.;
        }
    }
    pieces.extend(prepared.extras.into_iter().map(|s| (s, 0.)));
    *model =
        Unigram::from(pieces, model.unk_id(), model.byte_fallback())?.with_objective(objective);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainers::{MinGramTrainer, UnigramTrainer};
    #[test]
    fn numeric_only_training_and_fixed_budget() {
        for policy in [
            FixedDigits::single(false),
            FixedDigits::single(true),
            FixedDigits::triples(),
        ] {
            let count = policy.inventory().len() as u32;
            let mut model = Unigram::default();
            MinGramTrainer::builder()
                .show_progress(false)
                .alphabet_mode(AlphabetMode::Legacy)
                .fixed_digits(Some(policy))
                .vocab_size(count)
                .build()
                .unwrap()
                .do_train(vec![("1234567".into(), 5)], &mut model)
                .unwrap();
            assert_eq!(model.len(), count as usize);
            assert!(model.vocab().iter().all(|(_, s)| *s == 0.));
            UnigramTrainer::builder()
                .show_progress(false)
                .alphabet_mode(AlphabetMode::Legacy)
                .fixed_digits(Some(policy))
                .vocab_size(count)
                .build()
                .unwrap()
                .do_train(vec![("0001234".into(), 5)], &mut model)
                .unwrap();
            assert_eq!(model.len(), count as usize);
            assert!(model.vocab().iter().all(|(_, s)| *s == 0.));
            assert!(
                MinGramTrainer::builder()
                    .show_progress(false)
                    .alphabet_mode(AlphabetMode::Legacy)
                    .fixed_digits(Some(policy))
                    .vocab_size(count - 1)
                    .build()
                    .unwrap()
                    .do_train(vec![("1234".into(), 5)], &mut model)
                    .is_err()
            );
        }
    }
    #[test]
    fn fixed_policy_round_trips_in_trainer_settings() {
        let trainer = MinGramTrainer::builder()
            .fixed_digits(Some(FixedDigits::triples()))
            .build()
            .unwrap();
        let read: MinGramTrainer =
            serde_json::from_str(&serde_json::to_string(&trainer).unwrap()).unwrap();
        assert_eq!(read.fixed_digits, trainer.fixed_digits);
    }
}
