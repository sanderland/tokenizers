use crate::error::err;
use crate::tokenizer::Tokenizer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;
use tk::Trainer;
use tk::decoders::{
    DecoderRuntime, byte_fallback::ByteFallback, byte_level::ByteLevelDecoder, fuse::Fuse,
    unmark::Unmark,
};
use tk::models::bpe::{BpeConfig, PipelineBPE};
use tk::models::unigram::Unigram;
use tk::normalizers::{ByteLevel, ClaudeNormalizer, MarkBoundaries};
use tk::pipeline::{Normalizer as _, PreTokenizer as _};
use tk::pipeline::{PipelineModel, PipelinePostProcessor, PipelinePreTokenizer, PipelineTokenizer};
use tk::pre_tokenizers::{mark_split::MarkSplit, sequence::PipelineSequence};
use tk::trainers::{
    AlphabetMode, BpeTrainer, MinGramTrainer, PruneCriterion, SeedInit, UnigramTrainer,
};
use tk::vocab::bucket_added_vocabulary::{AddedToken, AddedVocabulary};

#[pyfunction]
#[pyo3(signature = (kind, sequences, vocab_size, show_progress, special_tokens, seed_init, seed_size, max_piece_length, n_sub_iterations, shrinking_factor, seed_factor, prune_criterion, pruning_shrinking_factor, skip_substring_in_batch, unk_token, min_frequency, initial_alphabet, alphabet_mode, fixed_digits, recipe, template))]
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
    recipe: Option<&str>,
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
    let digits = match fixed_digits {
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
    if let Some(name) = recipe {
        if !matches!(name, "claude_fixed" | "claude_v3" | "claude_v47") {
            return Err(PyValueError::new_err("unknown recipe"));
        }
        if kind == "bpe" || byte_level {
            return Err(PyValueError::new_err(
                "Claude recipes require a Unigram or MinGram trainer with utf8 or legacy alphabet",
            ));
        }
    }
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
    let mut template = template
        .map(|t| t.training_template())
        .transpose()?
        .flatten();
    if template.is_some() && recipe.is_some() {
        return Err(PyValueError::new_err("recipe requires a new tokenizer"));
    }
    if template.is_some() && digits.is_some() {
        return Err(PyValueError::new_err(
            "fixed_digits requires a new tokenizer; existing preprocessing cannot safely be combined",
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
    if let Some(name) = recipe {
        let mut marking = match name {
            "claude_fixed" => MarkBoundaries::default(),
            "claude_v3" => MarkBoundaries::v3(),
            "claude_v47" => MarkBoundaries::v47(),
            _ => unreachable!(),
        };
        if let Some(policy) = digits {
            marking = marking.with_fixed_digits(policy).map_err(err)?;
        }
        let mut splitters = vec![PipelinePreTokenizer::MarkSplit(MarkSplit)];
        if let Some(policy) = digits {
            splitters.push(PipelinePreTokenizer::FixedDigits(policy));
        }
        template = Some(PipelineTokenizer::from_parts(
            AddedVocabulary::new(),
            vec![
                tk::pipeline::PipelineNormalizer::ClaudeNormalizer(ClaudeNormalizer::new(
                    name != "claude_v47",
                )),
                tk::pipeline::PipelineNormalizer::MarkBoundaries(marking),
            ],
            PipelinePreTokenizer::Sequence(PipelineSequence::new(splitters)),
            PipelineModel::Unigram(Unigram::default()),
            PipelinePostProcessor::default(),
            Some(DecoderRuntime::Unmark(Unmark::new())),
            BTreeMap::new(),
            None,
            None,
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
                let mut t = BpeTrainer::builder()
                    .vocab_size(vocab_size as usize)
                    .show_progress(show_progress)
                    .special_tokens(special.clone())
                    .min_frequency(min_frequency)
                    .initial_alphabet(bpe_alphabet)
                    .build();
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
                    unk_token,
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
        return Ok(Tokenizer::from_pipeline(
            template
                .with_model_and_added_vocabulary(model, added)
                .map_err(err)?,
        ));
    }
    Ok(Tokenizer::from_pipeline(PipelineTokenizer::from_parts(
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
    )))
}
