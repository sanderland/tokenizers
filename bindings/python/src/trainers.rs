use crate::error::err;
use crate::tokenizer::Tokenizer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;
use tk::Trainer;
use tk::decoders::{
    DecoderRuntime, byte_fallback::ByteFallback, byte_level::ByteLevelDecoder, fuse::Fuse,
};
use tk::models::bpe::{BpeConfig, PipelineBPE};
use tk::models::unigram::Unigram;
use tk::normalizers::ByteLevel;
use tk::pipeline::{Normalizer as _, PreTokenizer as _};
use tk::pipeline::{PipelineModel, PipelinePostProcessor, PipelinePreTokenizer, PipelineTokenizer};
use tk::trainers::{
    AlphabetMode, BpeTrainer, MinGramTrainer, PruneCriterion, SeedInit, UnigramTrainer,
};
use tk::vocab::bucket_added_vocabulary::{AddedToken, AddedVocabulary};

#[pyfunction]
#[pyo3(signature = (kind, sequences, vocab_size, show_progress, special_tokens, seed_init, seed_size, max_piece_length, n_sub_iterations, shrinking_factor, seed_factor, prune_criterion, pruning_shrinking_factor, skip_substring_in_batch, unk_token, min_frequency, initial_alphabet, alphabet_mode, fixed_digits, template))]
#[allow(clippy::too_many_arguments)]
pub fn train(
    py: Python<'_>,
    kind: &str,
    sequences: Vec<String>,
    vocab_size: u32,
    show_progress: bool,
    special_tokens: Vec<String>,
    seed_init: &str,
    seed_size: Option<usize>,
    max_piece_length: usize,
    n_sub_iterations: u32,
    shrinking_factor: f64,
    seed_factor: f64,
    prune_criterion: &str,
    pruning_shrinking_factor: f64,
    skip_substring_in_batch: bool,
    unk_token: Option<String>,
    min_frequency: u64,
    initial_alphabet: Vec<char>,
    alphabet_mode: &str,
    fixed_digits: Option<&str>,
    template: Option<PyRef<'_, Tokenizer>>,
) -> PyResult<Tokenizer> {
    if sequences.iter().all(|s| s.is_empty()) {
        return Err(PyValueError::new_err(
            "training corpus must contain nonempty text",
        ));
    }
    let alphabet = match alphabet_mode {
        "legacy" => AlphabetMode::Legacy,
        "utf8" => AlphabetMode::Utf8,
        "byte_level" => AlphabetMode::ByteLevel,
        _ => {
            return Err(PyValueError::new_err(
                "alphabet_mode must be legacy, utf8, or byte_level",
            ));
        }
    };
    let mut digits = match fixed_digits {
        None => None,
        Some("single") => Some(tk::pre_tokenizers::fixed_digits::FixedDigits::single(false)),
        Some("single_marked") => Some(tk::pre_tokenizers::fixed_digits::FixedDigits::single(true)),
        Some("triples") => Some(tk::pre_tokenizers::fixed_digits::FixedDigits::triples()),
        _ => {
            return Err(PyValueError::new_err(
                "fixed_digits must be single, single_marked, triples, or None",
            ));
        }
    };
    if kind == "bpe" && digits.is_some() {
        return Err(PyValueError::new_err(
            "fixed_digits requires a Unigram or MinGram trainer",
        ));
    }
    if kind == "bpe" && alphabet_mode == "utf8" {
        return Err(PyValueError::new_err(
            "utf8 alphabet_mode requires a Unigram or MinGram trainer",
        ));
    }
    let byte_level = alphabet_mode == "byte_level";
    let seed = match seed_init {
        "substrings" => SeedInit::Substrings,
        "bpe" => SeedInit::Bpe,
        _ => {
            return Err(PyValueError::new_err(
                "seed_init must be 'substrings' or 'bpe'",
            ));
        }
    };
    let prune = match prune_criterion {
        "usage_count" => PruneCriterion::UsageCount,
        "minimum_increase" => PruneCriterion::MinimumIncrease,
        _ => return Err(PyValueError::new_err("invalid prune_criterion")),
    };
    if max_piece_length == 0
        || n_sub_iterations == 0
        || !(0.0..1.0).contains(&shrinking_factor)
        || !(0.0..1.0).contains(&pruning_shrinking_factor)
        || !seed_factor.is_finite()
        || seed_factor < 1.0
    {
        return Err(PyValueError::new_err("invalid training parameters"));
    }
    let special: Vec<_> = special_tokens
        .into_iter()
        .map(|s| AddedToken::from(s, true))
        .collect();
    let source_model = template.as_ref().and_then(|t| t.model_config());
    let template = template
        .map(|t| t.training_template())
        .transpose()?
        .flatten();
    if let Some(pipeline) = &template {
        fn collect<'a>(p: &'a PipelinePreTokenizer, out: &mut Vec<&'a PipelinePreTokenizer>) {
            match p {
                PipelinePreTokenizer::Sequence(seq) => {
                    for child in seq.pre_tokenizers() {
                        collect(child, out);
                    }
                }
                PipelinePreTokenizer::None => (),
                _ => out.push(p),
            }
        }
        let mut splitters = Vec::new();
        collect(pipeline.get_pre_tokenizer(), &mut splitters);
        let found: Vec<_> = splitters
            .iter()
            .enumerate()
            .filter_map(|(i, p)| match p {
                PipelinePreTokenizer::FixedDigits(policy) => Some((i, *policy)),
                _ => None,
            })
            .collect();
        if !found.is_empty() {
            if found.len() != 1
                || found[0].0 + 1 != splitters.len()
                || splitters[..found[0].0]
                    .iter()
                    .any(|p| !matches!(p, PipelinePreTokenizer::MarkSplit(_)))
            {
                return Err(PyValueError::new_err(
                    "fixed digit splitter must follow only MarkSplit and be the last splitter",
                ));
            }
            if digits.is_some_and(|requested| requested != found[0].1) {
                return Err(PyValueError::new_err(
                    "trainer fixed_digits does not match the configured digit splitter",
                ));
            }
            digits = Some(found[0].1);
        } else if digits.is_some() {
            return Err(PyValueError::new_err(
                "fixed_digits requires a matching configured digit splitter",
            ));
        }
        for normalizer in pipeline.get_normalizers() {
            if let tk::pipeline::PipelineNormalizer::MarkBoundaries(marking) = normalizer {
                if marking.fixed_digits != digits {
                    return Err(PyValueError::new_err(
                        "MarkBoundaries and digit splitter policies must match",
                    ));
                }
                if byte_level {
                    return Err(PyValueError::new_err(
                        "byte_level is incompatible with MarkBoundaries",
                    ));
                }
            }
        }
        if byte_level
            && splitters
                .iter()
                .any(|p| matches!(p, PipelinePreTokenizer::MarkSplit(_)))
        {
            return Err(PyValueError::new_err(
                "byte_level is incompatible with MarkSplit",
            ));
        }
        if kind != "bpe"
            && alphabet_mode == "legacy"
            && matches!(pipeline.get_decoder(), Some(DecoderRuntime::Unmark(_)))
        {
            return Err(PyValueError::new_err(
                "Unmark pipelines require alphabet_mode=\"utf8\"",
            ));
        }
    }
    if kind == "bpe" && digits.is_some() {
        return Err(PyValueError::new_err(
            "fixed_digits requires a Unigram or MinGram trainer",
        ));
    }
    if byte_level && fixed_digits == Some("single_marked") {
        return Err(PyValueError::new_err(
            "single_marked digit boundaries are not compatible with byte_level normalization",
        ));
    }
    if byte_level
        && template.as_ref().is_some_and(|p| {
            !p.get_normalizers()
                .iter()
                .any(|n| matches!(n, tk::pipeline::PipelineNormalizer::ByteLevel(_)))
        })
    {
        return Err(PyValueError::new_err(
            "byte_level training requires an existing ByteLevel normalizer",
        ));
    }
    let preprocessing = template.clone();
    let mut added_special = template
        .as_ref()
        .map(|p| {
            let mut entries: Vec<_> = p
                .get_added_vocabulary()
                .get_added_tokens_decoder()
                .into_iter()
                .collect();
            entries.sort_by_key(|(id, _)| *id);
            entries
                .into_iter()
                .map(|(_, token)| token)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for token in &special {
        if !added_special
            .iter()
            .any(|existing| existing.content == token.content)
        {
            added_special.push(token.clone());
        }
    }
    let trained_options = source_model.clone();
    let (model, ids, size, decoder) = py.detach(move || -> PyResult<_> {
        let process = |s: &str| {
            let mut remaining = s;
            let mut parts = Vec::new();
            loop {
                let found = special
                    .iter()
                    .filter(|t| !t.content.is_empty())
                    .filter_map(|t| remaining.find(&t.content).map(|i| (i, t.content.len())))
                    .min_by_key(|&(i, len)| (i, std::cmp::Reverse(len)));
                let (text, rest) = match found {
                    Some((i, len)) => (&remaining[..i], Some(&remaining[i + len..])),
                    None => (remaining, None),
                };
                if !text.is_empty() {
                    match &preprocessing {
                        Some(pipeline) => parts.extend(pipeline.preprocess_for_training(text)?),
                        None => {
                            let normalized = if byte_level {
                                ByteLevel::new().normalize(text, 0)?.into_owned()
                            } else {
                                text.to_owned()
                            };
                            if let Some(splitter) = digits {
                                let mut spans = Vec::new();
                                splitter.pre_tokenize(
                                    &normalized,
                                    &mut Default::default(),
                                    &mut spans,
                                )?;
                                parts.extend(
                                    spans
                                        .into_iter()
                                        .map(|span| normalized[span.range()].to_owned()),
                                );
                            } else {
                                parts.push(normalized);
                            }
                        }
                    }
                }
                match rest {
                    Some(s) => remaining = s,
                    None => break,
                }
            }
            Ok(parts)
        };
        match kind {
            "bpe" => {
                let mut bpe_alphabet: std::collections::HashSet<char> =
                    initial_alphabet.iter().copied().collect();
                if byte_level {
                    let allowed: std::collections::HashSet<char> =
                        tk::utils::byte_level::utf8_byte_alphabet().collect();
                    if !bpe_alphabet.is_subset(&allowed) {
                        return Err(PyValueError::new_err(
                            "byte_level initial_alphabet must contain mapped UTF-8 bytes",
                        ));
                    }
                    bpe_alphabet.extend(allowed);
                    if (vocab_size as usize) < bpe_alphabet.len() + special.len() {
                        return Err(PyValueError::new_err(
                            "vocab_size is too small for the byte alphabet and special tokens",
                        ));
                    }
                }
                let config = source_model
                    .as_ref()
                    .filter(|m| m.get("type").and_then(|v| v.as_str()) == Some("BPE"));
                let option = |key: &str| {
                    config
                        .and_then(|m| m.get(key))
                        .and_then(|v| v.as_str())
                        .map(str::to_owned)
                };
                let prefix = option("continuing_subword_prefix");
                let suffix = option("end_of_word_suffix");
                let model_unk = option("unk_token").or(unk_token);
                let mut builder = BpeTrainer::builder()
                    .vocab_size(vocab_size as usize)
                    .show_progress(show_progress)
                    .special_tokens(special.clone())
                    .min_frequency(min_frequency)
                    .initial_alphabet(bpe_alphabet);
                if let Some(value) = &prefix {
                    builder = builder.continuing_subword_prefix(value.clone());
                }
                if let Some(value) = &suffix {
                    builder = builder.end_of_word_suffix(value.clone());
                }
                let mut t = builder.build();
                t.feed(sequences.into_iter(), process).map_err(err)?;
                let (vocab, merges, _) = t.train_vocab().map_err(err)?;
                let ids = special
                    .iter()
                    .filter_map(|s| {
                        vocab
                            .get(s.content.as_str())
                            .map(|id| (s.content.clone(), *id))
                    })
                    .collect::<BTreeMap<_, _>>();
                let size = vocab.len();
                let model = PipelineBPE::from_config(BpeConfig {
                    vocab,
                    merges,
                    unk_token: model_unk,
                    continuing_subword_prefix: prefix,
                    end_of_word_suffix: suffix,
                    fuse_unk: config
                        .and_then(|m| m.get("fuse_unk"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    byte_fallback: config
                        .and_then(|m| m.get("byte_fallback"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    ignore_merges: config
                        .and_then(|m| m.get("ignore_merges"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    byte_level: config
                        .and_then(|m| m.get("byte_level"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    dropout: config
                        .and_then(|m| m.get("dropout"))
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32),
                    ..Default::default()
                })
                .map_err(err)?;
                Ok((
                    PipelineModel::BPE(model),
                    ids,
                    size,
                    Some(if byte_level {
                        DecoderRuntime::ByteLevel(ByteLevelDecoder::new())
                    } else {
                        DecoderRuntime::Fuse(Fuse::new())
                    }),
                ))
            }
            "unigram" => {
                let mut b = UnigramTrainer::builder();
                b.vocab_size(vocab_size)
                    .show_progress(show_progress)
                    .special_tokens(special.clone())
                    .seed_init(seed)
                    .max_piece_length(max_piece_length)
                    .n_sub_iterations(n_sub_iterations)
                    .shrinking_factor(shrinking_factor)
                    .unk_token(unk_token);
                b.alphabet_mode(alphabet).fixed_digits(digits);
                b.initial_alphabet(initial_alphabet.iter().copied().collect());
                if let Some(n) = seed_size {
                    b.seed_size(n);
                }
                let mut t = b.build().unwrap();
                t.feed(sequences.into_iter(), process).map_err(err)?;
                let mut model = Unigram::default();
                t.train(&mut model).map_err(err)?;
                let ids = special
                    .iter()
                    .filter_map(|s| {
                        model
                            .token_to_id(&s.content)
                            .map(|id| (s.content.clone(), id))
                    })
                    .collect();
                let size = model.len();
                Ok((
                    PipelineModel::Unigram(model),
                    ids,
                    size,
                    Some(if byte_level {
                        DecoderRuntime::ByteLevel(ByteLevelDecoder::new())
                    } else {
                        DecoderRuntime::ByteFallback(ByteFallback::new())
                    }),
                ))
            }
            "mingram" => {
                let mut b = MinGramTrainer::builder();
                b.vocab_size(vocab_size)
                    .show_progress(show_progress)
                    .special_tokens(special.clone())
                    .seed_init(seed)
                    .max_piece_length(max_piece_length)
                    .n_sub_iterations(n_sub_iterations)
                    .seed_factor(seed_factor)
                    .prune_criterion(prune)
                    .pruning_shrinking_factor(pruning_shrinking_factor)
                    .skip_substring_in_batch(skip_substring_in_batch);
                b.alphabet_mode(alphabet).fixed_digits(digits);
                b.initial_alphabet(initial_alphabet.iter().copied().collect());
                if let Some(n) = seed_size {
                    b.seed_size(n);
                }
                let mut t = b.build().unwrap();
                t.feed(sequences.into_iter(), process).map_err(err)?;
                let mut model = Unigram::default();
                t.train(&mut model).map_err(err)?;
                let ids = special
                    .iter()
                    .filter_map(|s| {
                        model
                            .token_to_id(&s.content)
                            .map(|id| (s.content.clone(), id))
                    })
                    .collect();
                let size = model.len();
                Ok((
                    PipelineModel::Unigram(model),
                    ids,
                    size,
                    Some(if byte_level {
                        DecoderRuntime::ByteLevel(ByteLevelDecoder::new())
                    } else {
                        DecoderRuntime::ByteFallback(ByteFallback::new())
                    }),
                ))
            }
            _ => Err(PyValueError::new_err("unknown trainer")),
        }
    })?;
    let mut added = AddedVocabulary::new();
    added
        .add_special_tokens(
            added_special,
            size,
            |s| ids.get(s).copied().or_else(|| model.token_to_id(s)),
            None::<&tk::pipeline::PipelineNormalizer>,
        )
        .map_err(err)?;
    if let Some(template) = template {
        let pipeline = template
            .with_model_and_added_vocabulary(model, added)
            .map_err(err)?;
        return Tokenizer::from_trained_pipeline(pipeline, trained_options.as_deref());
    }
    Tokenizer::from_pipeline(PipelineTokenizer::from_parts(
        added,
        if byte_level {
            vec![tk::pipeline::PipelineNormalizer::ByteLevel(ByteLevel::new())]
        } else {
            vec![]
        },
        digits.map_or(
            PipelinePreTokenizer::None,
            PipelinePreTokenizer::FixedDigits,
        ),
        model,
        PipelinePostProcessor::default(),
        decoder,
        BTreeMap::new(),
        None,
        None,
    ))
}
