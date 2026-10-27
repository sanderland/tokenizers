//! Claude-style boundary normalizer: text -> ctok's marked stream.
//!
//! // EXISTING CODE (pre-hackathon): ported from ctok/ctok/normalize.py and
//! // ctok/ctok/constants.py. Message framing is intentionally excluded:
//! // no trailing-newline repricing, no head-space absorption, no message
//! // overhead. A normalizer rewrites text; it does not price messages.
//! // Text start is treated as message start for border purposes (`frame_bow`).
//!
//! This lives in the normalizer slot rather than with the pre-tokenizers for a
//! mechanical reason: pipeline pre-tokenizers emit spans into the text they are
//! given, while this is a rewrite (it inserts marker codepoints, deletes seam
//! spaces, and lowercases spans). The model then tiles the marked stream whole,
//! with the pre-tokenizer slot left empty. Offsets past this point are relative
//! to the marked stream.

use std::borrow::Cow;

use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::canonical_combining_class;

use crate::{pipeline, tokenizer::Result};

// ---- marker notation (ctok/ctok/constants.py) ----

const BOW_G: char = '\u{FDD0}';
const EOW_G: char = '\u{FDD1}';
const PAD_G: char = '\u{FDD2}';
const SHIFT_G: char = '\u{FDD3}';
const CAPS_G: char = '\u{FDD4}';
const BOW_S: &str = "\u{FDD0}";
const EOW_S: &str = "\u{FDD1}";

// Literal occurrences of the marker codepoints escape to BMP private use
// (ctok strips PUA input in NFC first, so these cannot collide with text).
const LITERAL_ESCAPES: [(char, char); 5] = [
    (BOW_G, '\u{E000}'),
    (EOW_G, '\u{E001}'),
    (PAD_G, '\u{E002}'),
    (SHIFT_G, '\u{E003}'),
    (CAPS_G, '\u{E004}'),
];

// ---- measured character tables (ctok/ctok/constants.py) ----

// Enumerated: no category rule reproduces this set.
const PUNCT_SYMS: &str = "—»«•°„–−£§€…√→（№†└│།·─═█";
// Category-So members of CJK Symbols and Punctuation: measured markerless.
const IDEOGRAPHIC_SYMBOLS: &str = "〄〒〓〠〶〷〾〿";
// Symbol-letters taking the full word model. Enumerated blocks.
const SYMBOL_LETTERS: [(u32, u32); 4] = [
    (0x16EE, 0x16F0),
    (0x2160, 0x2188),
    (0x24B6, 0x24E9),
    (0xA6E6, 0xA6EF),
];
const VARIATION_SELECTORS: (u32, u32) = (0xFE00, 0xFE0F);
const CONTRACTION_SUFFIXES: [&str; 7] = ["s", "t", "d", "m", "ll", "re", "ve"];
// Space separators folding to U+0020 (all Zs except U+3000, plus Zl/Zp).
// NUL folds to a space in its own step, as in ctok.
const FUNNY_SPACES: [char; 17] = [
    '\u{A0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}',
    '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200A}', '\u{2028}', '\u{2029}', '\u{202F}',
    '\u{205F}',
];
// Unicode 16.0 case pairs the models know (upper -> lower).
const NEW_CASE_PAIRS: [(char, char); 2] = [('\u{1C89}', '\u{1C8A}'), ('\u{A7CB}', '\u{264}')];
// Uncased for the models, though Unicode calls U+03F4 a capital.
const UNCASED_THETA: char = '\u{3F4}';

fn is_alphabetic(c: char) -> bool {
    c.is_alphabetic()
}

/// `\p{Cn}`: a `char` is unassigned when it sits in none of the seven major
/// groups. (`is_other` covers Cc, Cf and Co only; surrogates cannot be `char`.)
fn is_unassigned(c: char) -> bool {
    !(c.is_other()
        || c.is_letter()
        || c.is_mark()
        || c.is_number()
        || c.is_punctuation()
        || c.is_symbol()
        || c.is_separator())
}

/// The stream class of one codepoint (ctok `normalize.classify`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Wordy,
    Hard,
    Digit,
    Punct,
    Space,
    StrayMark,
}

fn is_hard_cp(o: u32) -> bool {
    o >= 0x10000
        || (0x4E00..=0x9FFF).contains(&o)
        || (0x3400..=0x4DBF).contains(&o)
        || (0xF900..=0xFAFF).contains(&o)
        || (0xAC00..=0xD7A3).contains(&o)
        || matches!(
            o,
            0x3005 | 0x3006 | 0x3031 | 0x3032 | 0x3033 | 0x3034 | 0x3035 | 0x303B | 0x303C
        )
}

fn is_new_cased(c: char) -> bool {
    NEW_CASE_PAIRS.iter().any(|&(hi, lo)| hi == c || lo == c)
}

fn classify(c: char, marks_part_of_words: bool) -> Class {
    if is_new_cased(c) {
        return Class::Wordy;
    }
    let o = c as u32;
    if c.is_separator_space() || c.is_separator_line() || c.is_separator_paragraph() {
        return Class::Space;
    }
    if matches!(c, '\t' | '\n' | '\r' | '\x0C' | '\x0B') {
        return Class::Space;
    }
    if SYMBOL_LETTERS
        .iter()
        .any(|&(lo, hi)| (lo..=hi).contains(&o))
    {
        return Class::Wordy;
    }
    if c.is_number_decimal_digit()
        && (o < 0x80 || (0x0660..=0x0669).contains(&o) || (0x06F0..=0x06F9).contains(&o))
    {
        return Class::Digit;
    }
    if o < 0x80 && (c.is_punctuation() || c.is_symbol()) {
        return Class::Punct;
    }
    if PUNCT_SYMS.contains(c) {
        return Class::Punct;
    }
    if (VARIATION_SELECTORS.0..=VARIATION_SELECTORS.1).contains(&o) {
        return Class::Hard;
    }
    if c.is_mark() {
        // Fixed mode keeps every mark in its word. Buggy mode (Claude) keeps
        // only Alphabetic marks in range; the rest split the word. Variation
        // selectors return Hard above and ride their base in both modes.
        return if marks_part_of_words || (o < 0x10000 && is_alphabetic(c)) {
            Class::Wordy
        } else {
            Class::Hard
        };
    }
    if c.is_letter() && !is_hard_cp(o) {
        return Class::Wordy;
    }
    Class::Hard
}

// ---- case (ctok `normalize.mark_case`) ----

fn lower_char(c: char) -> String {
    if c == UNCASED_THETA {
        return c.to_string();
    }
    if let Some(&(_, lo)) = NEW_CASE_PAIRS.iter().find(|&&(hi, _)| hi == c) {
        return lo.to_string();
    }
    c.to_lowercase().collect()
}

fn is_upper(c: char) -> bool {
    if c == UNCASED_THETA {
        return false;
    }
    if NEW_CASE_PAIRS.iter().any(|&(hi, _)| hi == c) {
        return true;
    }
    c.is_uppercase() && lower_char(c) != c.to_string()
}

fn is_lower(c: char) -> bool {
    NEW_CASE_PAIRS.iter().any(|&(_, lo)| lo == c) || c.is_lowercase()
}

fn span_is_upper(span: &[char]) -> bool {
    let cased: Vec<char> = span
        .iter()
        .copied()
        .filter(|&c| is_upper(c) || is_lower(c))
        .collect();
    !cased.is_empty() && cased.iter().all(|&c| is_upper(c))
}

/// Unicode "cased" for the final-sigma rule (Lu/Ll/Lt; close enough to
/// CPython's rule for spans the case markers reach).
fn is_cased(c: char) -> bool {
    c.is_letter_uppercase() || c.is_letter_lowercase() || c.is_letter_titlecase()
}

/// Python-compatible span lowering. `final_sigma` applies the Σ -> ς rule
/// (cased before, uncased after); without it Σ lowers to σ everywhere.
fn lower_span(span: &[char], final_sigma: bool) -> String {
    let mut out = String::new();
    for (i, &c) in span.iter().enumerate() {
        if final_sigma && c == 'Σ' {
            let prev_cased = i > 0 && is_cased(span[i - 1]);
            let next_cased = i + 1 < span.len() && is_cased(span[i + 1]);
            out.push(if prev_cased && !next_cased {
                'ς'
            } else {
                'σ'
            });
        } else {
            out.push_str(&lower_char(c));
        }
    }
    out
}

fn mark_case(span: &str, allcaps_min: Option<usize>, head_mark: bool) -> String {
    if head_mark {
        return span.to_string();
    }
    let chars: Vec<char> = span.chars().collect();
    if chars.is_empty() {
        return String::new();
    }
    if chars.contains(&'\u{1E9E}') {
        // ẞ blocks everywhere.
        return span.to_string();
    }
    if chars.contains(&'\u{130}') {
        // İ is transparent in the tail, blocking only at the head.
        let tail = &chars[1..];
        if is_upper(chars[0])
            && chars[0] != '\u{130}'
            && !tail.iter().any(|&c| c != '\u{130}' && is_upper(c))
        {
            let mut out = String::new();
            out.push(SHIFT_G);
            out.push_str(&lower_char(chars[0]));
            for &c in tail {
                if c == '\u{130}' {
                    out.push(c);
                } else {
                    out.push_str(&lower_char(c));
                }
            }
            return out;
        }
        return span.to_string();
    }
    let unlowerable = chars.iter().any(|&c| {
        (is_new_cased(c) || c.is_letter() || c.is_mark())
            && !is_lower(c)
            && (!is_upper(c) || lower_char(c) == c.to_string())
    });
    if allcaps_min.is_some_and(|m| span_is_upper(&chars) && chars.len() >= m) && !unlowerable {
        let mut out = String::new();
        out.push(CAPS_G);
        out.push_str(&lower_span(&chars, false).replace('ς', "σ"));
        return out;
    }
    if lower_char(chars[0]) == chars[0].to_string() {
        return span.to_string();
    }
    if is_upper(chars[0]) && !chars[1..].iter().any(|&c| is_upper(c)) {
        let mut out = String::new();
        out.push(SHIFT_G);
        out.push_str(&lower_span(&chars, true));
        return out;
    }
    span.to_string()
}

// ---- normalization (ctok `normalize.nfc`) ----

fn is_strip_control(c: char) -> bool {
    matches!(
        c as u32,
        0x01..=0x08 | 0x0B..=0x1F | 0x7F | 0x80..=0x9F
    )
}

fn nfc(text: &str, fold_quotes: bool) -> String {
    // Rust strings cannot hold surrogates; that step is a no-op here.
    let composed: String = text.nfc().collect();
    let mut out: String = composed.chars().filter(|&c| !is_strip_control(c)).collect();
    if out.contains('\0') {
        out = out.replace('\0', " ");
    }
    // Thai SARA AM composes; Lao and other compatibility chars do not fold.
    if out.contains("\u{0E4D}\u{0E32}") {
        out = out.replace("\u{0E4D}\u{0E32}", "\u{0E33}");
    }
    out = out
        .chars()
        .filter(|&c| !(0xE000..=0xF8FF).contains(&(c as u32)))
        .collect();
    if fold_quotes {
        out = out
            .chars()
            .map(|c| match c {
                '\u{2018}' | '\u{2019}' => '\'',
                '\u{201C}' | '\u{201D}' => '"',
                _ => c,
            })
            .collect();
    }
    if out.chars().any(|c| FUNNY_SPACES.contains(&c)) {
        out = out
            .chars()
            .map(|c| if FUNNY_SPACES.contains(&c) { ' ' } else { c })
            .collect();
    }
    out
}

// ---- runs (ctok `normalize._runs` and its predicates) ----

fn is_selector(ch: char) -> bool {
    (VARIATION_SELECTORS.0..=VARIATION_SELECTORS.1).contains(&(ch as u32))
}

fn is_separator(c: char) -> bool {
    (c as u32) < 0x10000 && c.is_mark() && !is_selector(c) && !is_alphabetic(c)
}

fn is_syriac_vowel(c: char) -> bool {
    let o = c as u32;
    o == 0x0711 || (0x0730..=0x073F).contains(&o)
}

fn is_stray_mark(c: char) -> bool {
    if is_syriac_vowel(c) {
        return false;
    }
    canonical_combining_class(c) != 0 && !is_separator(c)
}

fn unescape_literal(ch: char) -> char {
    LITERAL_ESCAPES
        .iter()
        .find(|&&(_, esc)| esc == ch)
        .map(|&(glyph, _)| glyph)
        .unwrap_or(ch)
}

fn marks_like_punct(ch: char) -> bool {
    let ch = unescape_literal(ch);
    let o = ch as u32;
    if o >= 0x10000 {
        return false;
    }
    if is_separator(ch) {
        return true;
    }
    let ideographic_punct =
        (0x3001..=0x303F).contains(&o) && (ch.is_punctuation() || IDEOGRAPHIC_SYMBOLS.contains(ch));
    (ch.is_punctuation() || ch.is_symbol() || ch.is_other_format() || is_unassigned(ch))
        && !ideographic_punct
}

fn digit_border(ch: char) -> bool {
    (ch.is_number_other() || (ch.is_number_decimal_digit() && !ch.is_ascii()))
        && (ch as u32) < 0x10000
}

fn digit_kind(ch: char) -> u8 {
    if ch.is_number_decimal_digit() {
        0
    } else if ch.is_number_other() {
        1
    } else {
        2
    }
}

fn is_digit_run(body: &str) -> bool {
    let mut chars = body.chars();
    match chars.next() {
        None => false,
        Some(first) => {
            let kind = digit_kind(first);
            kind != 2 && chars.all(|c| digit_kind(c) == kind)
        }
    }
}

fn digit_bow(body: &str) -> bool {
    is_digit_run(body) && body.chars().next().is_some_and(digit_border)
}

fn digit_eow(body: &str) -> bool {
    is_digit_run(body) && body.chars().last().is_some_and(digit_border)
}

fn hard_bow(body: &str) -> bool {
    body.chars()
        .next()
        .is_some_and(|c| is_selector(c) || marks_like_punct(c))
}

fn hard_eow(body: &str) -> bool {
    body.chars()
        .last()
        .is_some_and(|c| is_selector(c) || marks_like_punct(c))
}

fn hard_kind(ch: char) -> u8 {
    if marks_like_punct(ch) {
        0 // punct
    } else if digit_border(ch) {
        1 // number
    } else {
        2 // letter
    }
}

type Run = (Class, String);

fn split_runs(norm: &str, marks_part_of_words: bool) -> Vec<Run> {
    let chars: Vec<char> = norm.chars().collect();
    if chars.is_empty() {
        return vec![];
    }
    let mut out: Vec<Run> = vec![];
    let mut cur = String::from(chars[0]);
    let mut cur_cls = classify(chars[0], marks_part_of_words);
    // Stray-mark splitting is part of the mark bug; fixed mode never splits it.
    if !marks_part_of_words && cur_cls == Class::Wordy && is_stray_mark(chars[0]) {
        cur_cls = Class::StrayMark;
    }
    for &ch in &chars[1..] {
        let c = classify(ch, marks_part_of_words);
        if (cur_cls == Class::StrayMark && c == Class::Wordy && is_stray_mark(ch)) || c == cur_cls {
            cur.push(ch);
        } else if !marks_part_of_words
            && c == Class::Wordy
            && is_stray_mark(ch)
            && cur_cls != Class::Wordy
        {
            out.push((cur_cls, std::mem::take(&mut cur)));
            cur.push(ch);
            cur_cls = Class::StrayMark;
        } else {
            out.push((cur_cls, std::mem::take(&mut cur)));
            cur.push(ch);
            cur_cls = c;
        }
    }
    out.push((cur_cls, cur));

    // A HARD run is not homogeneous; split where the character kind changes.
    // A variation selector never opens a sub-run: it rides its base.
    let mut split: Vec<Run> = vec![];
    for (cls, body) in out {
        if cls != Class::Hard || body.chars().count() == 1 {
            split.push((cls, body));
            continue;
        }
        let mut sub = String::new();
        let mut sub_kind = 2;
        for (i, ch) in body.chars().enumerate() {
            if i == 0 {
                sub.push(ch);
                sub_kind = hard_kind(ch);
            } else if is_selector(ch) || hard_kind(ch) == sub_kind {
                sub.push(ch);
            } else {
                split.push((cls, std::mem::take(&mut sub)));
                sub.push(ch);
                sub_kind = hard_kind(ch);
            }
        }
        split.push((cls, sub));
    }
    split
}

fn opens_word(runs: &[Run], i: usize) -> bool {
    runs[i].1 == "'"
        && i + 1 < runs.len()
        && matches!(runs[i + 1].0, Class::Wordy | Class::StrayMark)
}

fn takes_right_border(cls: Class, body: &str) -> bool {
    cls == Class::Punct
        || hard_eow(body)
        || (matches!(cls, Class::Digit | Class::Hard) && is_digit_run(body) && digit_eow(body))
}

fn contraction_seam(runs: &[Run], i: usize) -> bool {
    if !CONTRACTION_SUFFIXES.contains(&runs[i].1.as_str()) || i == 0 {
        return false;
    }
    let (prev_cls, prev_body) = &runs[i - 1];
    if *prev_cls != Class::Punct || prev_body != "'" {
        return false;
    }
    i < 2 || !takes_right_border(runs[i - 2].0, &runs[i - 2].1)
}

// ---- the marked stream (ctok `normalize.stream_norm`, minus framing) ----

/// The seam law: `⟨eow⟩` + single space + `⟨bow⟩` (case markers allowed
/// between the space and `⟨bow⟩`) loses the space. Non-overlapping, left to
/// right, and the char before `⟨eow⟩` must exist and not be a newline, which
/// is what ctok's `(.)⟨eow⟩ ([⟨shift⟩⟨caps⟩]*)⟨bow⟩` replace-all does.
fn apply_seam_law(marked: &str) -> String {
    let chars: Vec<char> = marked.chars().collect();
    let mut out = String::with_capacity(marked.len());
    let mut i = 0;
    while i < chars.len() {
        let seam =
            i > 0 && chars[i - 1] != '\n' && chars[i] == EOW_G && chars.get(i + 1) == Some(&' ');
        if !seam {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let mut j = i + 2;
        while matches!(chars.get(j).copied(), Some(SHIFT_G) | Some(CAPS_G)) {
            j += 1;
        }
        if chars.get(j) == Some(&BOW_G) {
            out.push(EOW_G);
            out.extend(&chars[i + 2..j]);
            out.push(BOW_G);
            i = j + 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Claude-style boundary normalizer: rewrites text to ctok's marked stream
/// (word boundaries, case and absorbed spaces as markers). The model then
/// tiles the marked stream whole.
///
/// Options select the ctok family boundary rules. The default is the fixed
/// configuration: v3 rules with the mark-splitting bug repaired.
/// `v3`, `v47` and `v48` reproduce ctok's families exactly, bug included.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Claude {
    /// Fold curly quotes to ASCII (v3 only).
    pub fold_quotes: bool,
    /// All-caps spans of at least this length take ⟨caps⟩; None disables.
    pub allcaps_min: Option<usize>,
    /// Text starts as if after a space (v3/v4.7; v4.8 has no frame ⟨bow⟩).
    pub frame_bow: bool,
    /// Keep combining marks inside their words. Claude instead splits words
    /// at marks without the Unicode Alphabetic property (combining accents,
    /// viramas, tone marks), so Yoruba, Hindi, Thai and friends tokenize
    /// badly; see "Small pre-tokenization bugs with a big multilingual
    /// price" (Token Contributions, Sep 2026). True is the fix and the
    /// default; the ctok family presets set it false to match Claude.
    pub marks_part_of_words: bool,
}

impl Default for Claude {
    /// v3 boundary rules with the mark-splitting bug repaired.
    fn default() -> Self {
        Self {
            fold_quotes: true,
            allcaps_min: Some(4),
            frame_bow: true,
            marks_part_of_words: true,
        }
    }
}

impl Claude {
    pub fn new(
        fold_quotes: bool,
        allcaps_min: Option<usize>,
        frame_bow: bool,
        marks_part_of_words: bool,
    ) -> Self {
        Self {
            fold_quotes,
            allcaps_min,
            frame_bow,
            marks_part_of_words,
        }
    }

    /// v3 boundary rules, exactly as ctok measures them (marks split words).
    pub fn v3() -> Self {
        Self {
            fold_quotes: true,
            allcaps_min: Some(4),
            frame_bow: true,
            marks_part_of_words: false,
        }
    }

    /// v4.7 boundary rules: no quote folding, no ⟨caps⟩ (marks split words).
    pub fn v47() -> Self {
        Self {
            fold_quotes: false,
            allcaps_min: None,
            frame_bow: true,
            marks_part_of_words: false,
        }
    }

    /// v4.8 boundary rules: v4.7 without the frame ⟨bow⟩ (marks split words).
    pub fn v48() -> Self {
        Self {
            fold_quotes: false,
            allcaps_min: None,
            frame_bow: false,
            marks_part_of_words: false,
        }
    }

    /// Text -> the marked stream, in ctok's internal glyph form.
    pub fn marked_stream(&self, text: &str) -> String {
        self.stream_norm(&nfc(text, self.fold_quotes))
    }

    fn stream_norm(&self, norm: &str) -> String {
        // Protect literal occurrences of the internal marker codepoints.
        let norm: String = norm
            .chars()
            .map(|c| {
                LITERAL_ESCAPES
                    .iter()
                    .find(|&&(glyph, _)| glyph == c)
                    .map(|&(_, esc)| esc)
                    .unwrap_or(c)
            })
            .collect();

        let runs = split_runs(&norm, self.marks_part_of_words);
        if runs.is_empty() {
            return if self.frame_bow {
                BOW_G.to_string()
            } else {
                String::new()
            };
        }
        let n_runs = runs.len();

        let borders_space = |i: usize, side: i32| -> bool {
            let j = i as i32 + side;
            if j < 0 {
                return self.frame_bow;
            }
            if j >= n_runs as i32 {
                return false;
            }
            let (cls, body) = &runs[j as usize];
            if *cls != Class::Space {
                return false;
            }
            if side < 0 {
                return body.ends_with(' ');
            }
            // A right-hand marker is written for the seam space only, never
            // before a run of two or more spaces.
            let mut chars = body.chars();
            chars.next() == Some(' ') && chars.next() != Some(' ')
        };

        let first = &runs[0];
        let head_quote = opens_word(&runs, 0);
        let has_own_bow = !head_quote
            && (matches!(first.0, Class::Wordy | Class::Punct | Class::StrayMark)
                || hard_bow(&first.1)
                || (matches!(first.0, Class::Digit | Class::Hard) && digit_bow(&first.1))
                || (first.0 == Class::Space && first.1.starts_with(' ')));
        let mut out: Vec<String> = if has_own_bow || !self.frame_bow {
            vec![]
        } else if head_quote {
            vec![" ".to_string()]
        } else {
            vec![BOW_G.to_string()]
        };

        for (i, (cls, body)) in runs.iter().enumerate() {
            match cls {
                Class::Wordy => {
                    let fused = i > 0 && runs[i - 1].0 == Class::StrayMark;
                    let marked = mark_case(body, self.allcaps_min, fused);
                    // Case markers precede ⟨bow⟩ in the file's spelling.
                    let mut chars = marked.chars();
                    let mut pre = String::new();
                    let mut rest = marked.as_str();
                    while matches!(chars.clone().next(), Some(SHIFT_G) | Some(CAPS_G)) {
                        let m = chars.next().unwrap();
                        pre.push(m);
                        rest = &rest[m.len_utf8()..];
                    }
                    let bow = if fused || contraction_seam(&runs, i) {
                        ""
                    } else {
                        BOW_S
                    };
                    out.push(format!("{pre}{bow}{rest}{EOW_S}"));
                }
                Class::StrayMark => {
                    let letter_follows = i + 1 < n_runs && runs[i + 1].0 == Class::Wordy;
                    out.push(format!(
                        "{BOW_S}{body}{}",
                        if letter_follows { "" } else { EOW_S }
                    ));
                }
                _ if *cls == Class::Punct || hard_bow(body) || hard_eow(body) => {
                    let takes_bow = borders_space(i, -1)
                        && !opens_word(&runs, i)
                        && (*cls == Class::Punct || hard_bow(body));
                    let takes_eow = borders_space(i, 1) && (*cls == Class::Punct || hard_eow(body));
                    out.push(format!(
                        "{}{}{}",
                        if takes_bow { BOW_S } else { "" },
                        body,
                        if takes_eow { EOW_S } else { "" }
                    ));
                }
                _ if is_digit_run(body) && matches!(cls, Class::Digit | Class::Hard) => {
                    let takes_bow = digit_bow(body) && borders_space(i, -1);
                    let takes_eow = digit_eow(body) && borders_space(i, 1);
                    out.push(format!(
                        "{}{}{}",
                        if takes_bow { BOW_S } else { "" },
                        body,
                        if takes_eow { EOW_S } else { "" }
                    ));
                }
                _ => out.push(body.clone()),
            }
        }

        apply_seam_law(&out.concat())
    }
}

impl pipeline::Normalizer for Claude {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        // Every input is its own message: text start counts as message start
        // for border purposes regardless of where the input sits in a pair.
        Ok(Cow::Owned(self.marked_stream(input)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ctok public notation: ⟨bow⟩=BOW_G, ⟨eow⟩=EOW_G, ⟨shift⟩=SHIFT_G,
    // ⟨caps⟩=CAPS_G. Expectations below were captured from ctok's
    // marked_stream on inputs without framing effects (no leading space,
    // no trailing newlines), where the normalizer must agree exactly.
    const B: &str = "\u{FDD0}";
    const E: &str = "\u{FDD1}";
    const S: &str = "\u{FDD3}";
    const C: &str = "\u{FDD4}";

    fn v3_cases() -> Vec<(&'static str, String)> {
        vec![
            ("hello", format!("{B}hello{E}")),
            ("Hello", format!("{S}{B}hello{E}")),
            ("HELLO", format!("{C}{B}hello{E}")),
            ("HEL", format!("{B}HEL{E}")),
            ("a!b", format!("{B}a{E}!{B}b{E}")),
            ("a! b", format!("{B}a{E}!{E}{B}b{E}")),
            ("hello world", format!("{B}hello{E}{B}world{E}")),
            ("it's", format!("{B}it{E}'s{E}")),
            ("123", format!("{B}123")),
            ("123 456", format!("{B}123 456")),
            ("a  b", format!("{B}a{E}  {B}b{E}")),
            ("don't stop", format!("{B}don{E}'t{E}{B}stop{E}")),
            ("NASA", format!("{C}{B}nasa{E}")),
            ("WiFi", format!("{B}WiFi{E}")),
            ("can't", format!("{B}can{E}'t{E}")),
            ("'tis", format!(" '{B}tis{E}")),
            ("x 'REXX", format!("{B}x{E} '{C}{B}rexx{E}")),
            ("a,b", format!("{B}a{E},{B}b{E}")),
            ("1,000", format!("{B}1,000")),
            ("e.g.", format!("{B}e{E}.{B}g{E}.")),
            ("Café", format!("{S}{B}café{E}")),
            ("Σίσυφος", format!("{S}{B}σίσυφος{E}")),
            ("ΟΔΥΣΣΕΥΣ", format!("{C}{B}οδυσσευσ{E}")),
            ("Straße", format!("{S}{B}straße{E}")),
            ("ẞtest", format!("{B}ẞtest{E}")),
            ("\u{201C}hi\u{201D}", format!("{B}\"{B}hi{E}\"")),
            ("a'b", format!("{B}a{E}'{B}b{E}")),
        ]
    }

    #[test]
    fn test_marked_stream_v3() {
        let claude = Claude::v3();
        for (input, expected) in v3_cases() {
            assert_eq!(claude.marked_stream(input), expected, "input {:?}", input);
        }
    }

    #[test]
    fn test_marks_part_of_words_keeps_marks_in_words() {
        // The article's three examples: Yoruba graves, Hindi viramas and the
        // Thai tone mark split words under the Claude presets but stay inside
        // under the default.
        let fixed = Claude::default();
        assert!(fixed.marks_part_of_words);
        assert_eq!(fixed.marked_stream("ọ̀rọ̀"), format!("{B}ọ̀rọ̀{E}"));
        assert_eq!(fixed.marked_stream("राष्ट्रीय"), format!("{B}राष्ट्रीय{E}"));
        // วันนี้ with the final U+0E49 tone mark spelled out: it is the mark
        // the bug splits off, so the literal must not lose it silently.
        assert_eq!(
            fixed.marked_stream("ว\u{e31}นน\u{e35}\u{e49}"),
            format!("{B}ว\u{e31}นน\u{e35}\u{e49}{E}")
        );

        let buggy = Claude::v3();
        assert!(!buggy.marks_part_of_words);
        assert_eq!(buggy.marked_stream("ọ̀rọ̀"), format!("{B}ọ{E}̀{B}rọ{E}̀"));
        assert_eq!(
            buggy.marked_stream("राष्ट्रीय"),
            format!("{B}राष{E}्{B}ट{E}्{B}रीय{E}")
        );
        assert_eq!(
            buggy.marked_stream("ว\u{e31}นน\u{e35}\u{e49}"),
            format!("{B}ว\u{e31}นน\u{e35}{E}\u{e49}")
        );
    }

    #[test]
    fn test_marks_part_of_words_keeps_vs_riding() {
        // A variation selector still rides its base in fixed mode: the emoji
        // stays one hard run, and the selector never opens a word run.
        let fixed = Claude::default();
        assert_eq!(fixed.marked_stream("a✈️b"), format!("{B}a{E}✈️{B}b{E}"));
    }

    #[test]
    fn test_marked_stream_v47() {
        let claude = Claude::v47();
        assert_eq!(claude.marked_stream("HELLO"), format!("{B}HELLO{E}"));
        assert_eq!(
            claude.marked_stream("\u{201C}hi\u{201D}"),
            format!("{B}\u{201C}{B}hi{E}\u{201D}")
        );
        assert_eq!(claude.marked_stream("Hello"), format!("{S}{B}hello{E}"));
        assert_eq!(claude.marked_stream("123"), format!("{B}123"));
    }

    #[test]
    fn test_marked_stream_v48() {
        let claude = Claude::v48();
        assert_eq!(claude.marked_stream("123"), "123");
        assert_eq!(claude.marked_stream("'tis"), format!("'{B}tis{E}"));
        assert_eq!(claude.marked_stream("hello"), format!("{B}hello{E}"));
    }

    #[test]
    fn test_marked_stream_empty() {
        assert_eq!(Claude::default().marked_stream(""), B);
        assert_eq!(Claude::v48().marked_stream(""), "");
    }

    #[test]
    fn test_normalize_rewrites() {
        use crate::pipeline::Normalizer;
        let claude = Claude::default();
        assert_eq!(
            &*claude.normalize("hello world", 0).unwrap(),
            format!("{B}hello{E}{B}world{E}")
        );
    }
}
