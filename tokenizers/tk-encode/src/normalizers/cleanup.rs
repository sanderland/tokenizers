//! Lossy cleanup normalizer: text -> ctok's `nfc` form.
//!
//! // EXISTING CODE (pre-hackathon): ported from ctok/ctok/normalize.py
//! // (`nfc`). This is the lossy half of the pipeline: NFC, control
//! // stripping, NUL/space/quote folding. It cannot be decoded back.
//! // Pair it with [`MarkBoundaries`](super::mark_boundaries::MarkBoundaries)
//! // in a `Sequence`: cleanup first, boundary marking second.
//!
//! Every step is idempotent, so running cleanup twice (once here, once
//! inside a composed call) changes nothing but costs a second pass.

use std::borrow::Cow;

use unicode_normalization::UnicodeNormalization;

use crate::{pipeline, tokenizer::Result};

// Space separators folding to U+0020 (all Zs except U+3000, plus Zl/Zp).
// NUL folds to a space in its own step, as in ctok.
const FUNNY_SPACES: [char; 17] = [
    '\u{A0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}',
    '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200A}', '\u{2028}', '\u{2029}', '\u{202F}',
    '\u{205F}',
];

fn is_strip_control(c: char) -> bool {
    matches!(
        c as u32,
        0x01..=0x08 | 0x0B..=0x1F | 0x7F | 0x80..=0x9F
    )
}

/// Lossy cleanup: NFC, control stripping, NUL/space/quote folding.
/// The v3 pairing uses `fold_quotes: true`, v4.7+ `false`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Cleanup {
    /// Fold curly quotes to ASCII (v3 only).
    pub fold_quotes: bool,
}

impl Default for Cleanup {
    /// v3 cleanup rules (quote folding on).
    fn default() -> Self {
        Self { fold_quotes: true }
    }
}

impl Cleanup {
    pub fn new(fold_quotes: bool) -> Self {
        Self { fold_quotes }
    }

    /// Text -> ctok's `nfc` form.
    pub fn clean(&self, text: &str) -> String {
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
        if self.fold_quotes {
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
}

impl pipeline::Normalizer for Cleanup {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        Ok(Cow::Owned(self.clean(input)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_folds_quotes_spaces_and_controls() {
        let clean = Cleanup::new(true);
        assert_eq!(clean.clean("\u{201C}hi\u{201D}"), "\"hi\"");
        assert_eq!(clean.clean("a\u{A0}b\u{2003}c"), "a b c");
        assert_eq!(clean.clean("a\x01b\x7F"), "ab");
        assert_eq!(clean.clean("a\0b"), "a b");
        assert_eq!(clean.clean("e\u{301}"), "é");
        assert_eq!(clean.clean("\u{0E4D}\u{0E32}"), "\u{0E33}");
        assert_eq!(clean.clean("a\u{E000}b"), "ab");
        assert_eq!(clean.clean("a\tb\nc"), "a\tb\nc");

        let unfolded = Cleanup::new(false);
        assert_eq!(unfolded.clean("\u{201C}hi\u{201D}"), "\u{201C}hi\u{201D}");
        assert_eq!(unfolded.clean("a\u{A0}b"), "a b");
    }

    #[test]
    fn test_clean_is_idempotent() {
        for text in ["\u{201C}a\u{A0}b\x01\0", "plain", ""] {
            let once = Cleanup::new(true).clean(text);
            assert_eq!(Cleanup::new(true).clean(&once), once);
        }
    }
}
