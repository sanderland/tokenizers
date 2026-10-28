//! Reverses [`MarkBoundaries`](crate::normalizers::MarkBoundaries): joined
//! pieces back to clean text.
//!
//! Three passes over the concatenated pieces:
//!
//! 1. Byte runs: whole tokens spelling `<0x..>` (singles and multibyte
//!    prefixes) back to text, mirroring [`ByteFallback`]: one `�` per piece
//!    when the run is not valid UTF-8. This runs first because decoded bytes
//!    can hold markers: unseen content fuses with adjacent unknown markers
//!    before tiling.
//! 2. Seams: `⟨eow⟩[⟨shift⟩⟨caps⟩]*⟨bow⟩` back to the space the seam law ate.
//!    Every such adjacency ate one: a `⟨bow⟩` directly after `⟨eow⟩` is only
//!    reachable through the seam, since the runs that take a free `⟨bow⟩`
//!    never directly follow a run that takes `⟨eow⟩`.
//! 3. Markers: `⟨bow⟩`/`⟨eow⟩` strip, `⟨shift⟩` uppercases the next char,
//!    `⟨caps⟩` uppercases to `⟨eow⟩`, `⟨pad⟩` strips, and the U+E000-U+E004
//!    literal escapes map back to the markers they protect.
//!
//! This inverts clean+mark, so `decode(mark(clean(x))) == clean(x)`; the
//! lossy cleanup (NFC, quote folding, control stripping) is one-way. Literal
//! `<0x..>` text that tiles as byte pieces decodes as bytes, the same
//! ambiguity [`ByteFallback`] accepts.
//!
//! [`ByteFallback`]: super::byte_fallback::ByteFallback
//! [`MarkBoundaries`]: crate::normalizers::MarkBoundaries

use super::Decoder;
use crate::tokenizer::Result;

const BOW_G: char = '\u{FDD0}';
const EOW_G: char = '\u{FDD1}';
const PAD_G: char = '\u{FDD2}';
const SHIFT_G: char = '\u{FDD3}';
const CAPS_G: char = '\u{FDD4}';

/// The decoder for [`MarkBoundaries`](crate::normalizers::MarkBoundaries)
/// streams, all option settings: case markers and seams are self-describing,
/// so one decoder inverts every configuration.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct Unmark {}

impl Unmark {
    pub fn new() -> Self {
        Self {}
    }
}

/// The bytes of a whole `<0x..>` token, or `None` when the token is not one.
/// Odd hex and non-hex spellings are content, not bytes.
fn parse_byte_token(token: &str) -> Option<Vec<u8>> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.is_empty() || hex.len() % 2 != 0 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok())
        .collect()
}

/// Join tokens, decoding maximal runs of byte tokens as one UTF-8 string.
/// A run that does not decode yields one `�` per piece, as [`ByteFallback`]
/// does.
///
/// [`ByteFallback`]: super::byte_fallback::ByteFallback
fn join_bytes(tokens: &[String]) -> String {
    let mut out = String::new();
    let mut run_bytes: Vec<u8> = vec![];
    let mut run_pieces = 0;
    let flush = |out: &mut String, run_bytes: &mut Vec<u8>, run_pieces: &mut usize| {
        if *run_pieces == 0 {
            return;
        }
        match String::from_utf8(std::mem::take(run_bytes)) {
            Ok(text) => out.push_str(&text),
            Err(_) => {
                for _ in 0..*run_pieces {
                    out.push('�');
                }
            }
        }
        *run_pieces = 0;
    };
    for token in tokens {
        match parse_byte_token(token) {
            Some(mut bytes) => {
                run_bytes.append(&mut bytes);
                run_pieces += 1;
            }
            None => {
                flush(&mut out, &mut run_bytes, &mut run_pieces);
                out.push_str(token);
            }
        }
    }
    flush(&mut out, &mut run_bytes, &mut run_pieces);
    out
}

/// The seam law backwards: `⟨eow⟩[⟨shift⟩⟨caps⟩]*⟨bow⟩` restores the eaten
/// space, keeping the case markers for the next pass. Only the `⟨bow⟩` goes:
/// the `⟨eow⟩` stays because it ends the `⟨caps⟩` extent, which the marker
/// pass reads.
fn restore_seams(marked: &str) -> String {
    let chars: Vec<char> = marked.chars().collect();
    let mut out = String::with_capacity(marked.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == EOW_G {
            let mut j = i + 1;
            while matches!(chars.get(j).copied(), Some(SHIFT_G) | Some(CAPS_G)) {
                j += 1;
            }
            if chars.get(j) == Some(&BOW_G) {
                out.push(EOW_G);
                out.push(' ');
                out.extend(&chars[i + 1..j]);
                i = j + 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Strip markers, applying case and unescaping literals. A `⟨shift⟩`
/// uppercases the next char, a `⟨caps⟩` everything to `⟨eow⟩`; both die at
/// `⟨eow⟩`, past which their unit cannot reach.
fn strip_markers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut shift = false;
    let mut caps = false;
    for c in text.chars() {
        match c {
            BOW_G | PAD_G => {}
            EOW_G => {
                shift = false;
                caps = false;
            }
            SHIFT_G => shift = true,
            CAPS_G => caps = true,
            '\u{E000}' => out.push(BOW_G),
            '\u{E001}' => out.push(EOW_G),
            '\u{E002}' => out.push(PAD_G),
            '\u{E003}' => out.push(SHIFT_G),
            '\u{E004}' => out.push(CAPS_G),
            _ => {
                if shift {
                    shift = false;
                    out.extend(c.to_uppercase());
                } else if caps {
                    out.extend(c.to_uppercase());
                } else {
                    out.push(c);
                }
            }
        }
    }
    out
}

impl Decoder for Unmark {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        Ok(vec![strip_markers(&restore_seams(&join_bytes(&tokens)))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::{ClaudeNormalizer, MarkBoundaries};

    fn decode(tokens: &[&str]) -> String {
        Unmark::new()
            .decode(tokens.iter().map(|t| t.to_string()).collect())
            .unwrap()
    }

    #[test]
    fn seams_spaces_and_case() {
        const B: &str = "\u{FDD0}";
        const E: &str = "\u{FDD1}";
        const S: &str = "\u{FDD3}";
        const C: &str = "\u{FDD4}";
        assert_eq!(decode(&[&format!("{B}a{E}{B}b{E}")]), "a b");
        assert_eq!(decode(&[&format!("{B}a{E}  {B}b{E}")]), "a  b");
        assert_eq!(decode(&[&format!("{B}a{E}\n{B}b{E}")]), "a\nb");
        assert_eq!(decode(&[&format!("{B}a{E}{S}{B}b{E}")]), "a B");
        assert_eq!(decode(&[&format!("{S}{B}hello{E}")]), "Hello");
        assert_eq!(decode(&[&format!("{C}{B}hello{E}")]), "HELLO");
        assert_eq!(decode(&[&format!("{B}don{E}'t{E}")]), "don't");
        assert_eq!(decode(&[&format!("{B}a{E},{E}{B}b{E}")]), "a, b");
        assert_eq!(decode(&[&format!("{B}a{E}{B},{B}b{E}")]), "a ,b");
        // A seam is always an eaten space, even before a combining mark:
        // splits without a space stand bare instead of taking the seam.
        assert_eq!(decode(&[&format!("{B}e{E}{B}\u{301}x{E}")]), "e \u{301}x");
        // Separators stand bare between units and pass through.
        assert_eq!(
            decode(&["\u{FDD0}น\u{FDD1}\u{e49}\u{FDD0}ำ\u{FDD1}"]),
            "น\u{e49}ำ"
        );
    }

    #[test]
    fn bytes_pads_escapes_and_passthrough() {
        assert_eq!(decode(&["\u{FDD0}", "<0x41>", "\u{FDD1}"]), "A");
        assert_eq!(decode(&["<0xE0>", "<0xB9>", "<0x88>"]), "\u{e48}");
        assert_eq!(decode(&["<0xE0B9>", "<0x88>"]), "\u{e48}");
        assert_eq!(decode(&["<0x41>", "bc"]), "Abc");
        assert_eq!(decode(&["<0xFF>"]), "�");
        assert_eq!(decode(&["<0xE0>", "<0x41>"]), "��");
        // Not whole byte tokens: content, left alone.
        assert_eq!(decode(&["<0x4>"]), "<0x4>");
        assert_eq!(decode(&["<0xGG>"]), "<0xGG>");
        assert_eq!(decode(&["a<0x41>b"]), "a<0x41>b");
        assert_eq!(decode(&["a\u{FDD2}b"]), "ab");
        assert_eq!(decode(&["\u{E000}x\u{E004}"]), "\u{FDD0}x\u{FDD4}");
        assert_eq!(decode(&["<unk>"]), "<unk>");
        assert_eq!(decode(&[""]), "");
    }

    fn battery() -> Vec<&'static str> {
        vec![
            "hello world",
            "Hello World",
            "HELLO WORLD",
            "HELLO world",
            "hello WORLD",
            "ABC",
            "ABCD",
            "don't stop",
            "a,b",
            "a, b",
            "a ,b",
            "(parens) [brackets]",
            "email@example.com",
            "  two leading",
            "trailing ",
            "a  b",
            "line1\nline2",
            "a\n\nb",
            "123 4567",
            "3.14",
            "1,000",
            "日本語のテスト",
            "🎉 party 🎉",
            "fn main() { println!(\"hi\"); }",
            "e\u{301}x",
            "e \u{301}x",
            "हिन्दी नमस्ते",
            "น้ำ ข้าว",
            "nai\u{308}ve cafe\u{301}",
            "Aİb",
            "ΟΔΥΣΣΕΎΣ",
            "",
            "   ",
            "a\u{FDD0}b",
            "a\u{E000}b",
        ]
    }

    /// Tiling is a partition: whole, halves, and thirds must decode alike.
    fn partitions(marked: &str) -> Vec<Vec<String>> {
        let chars: Vec<char> = marked.chars().collect();
        let span = |a: usize, b: usize| chars[a..b].iter().collect::<String>();
        let n = chars.len();
        vec![
            vec![marked.to_string()],
            vec![span(0, n / 2), span(n / 2, n)],
            vec![span(0, n / 3), span(n / 3, 2 * n / 3), span(2 * n / 3, n)],
        ]
    }

    #[test]
    fn decode_inverts_clean_mark() {
        let configs = [
            (ClaudeNormalizer::new(true), MarkBoundaries::v3()),
            (ClaudeNormalizer::new(false), MarkBoundaries::v47()),
            (ClaudeNormalizer::new(true), MarkBoundaries::default()),
        ];
        let decoder = Unmark::new();
        for (cleaner, marking) in &configs {
            for raw in battery() {
                let cleaned = cleaner.clean(raw);
                let marked = marking.marked_stream(&cleaned);
                for pieces in partitions(&marked) {
                    let back = decoder.decode(pieces.clone()).unwrap();
                    assert_eq!(back, cleaned, "raw {raw:?} marked {marked:?}");
                }
            }
        }
    }

    #[test]
    fn space_split_and_bare_split_stay_distinct() {
        // The spaced split takes the seam (bow after the space); the bare
        // split stands markerless. The shapes differ, so both round-trip.
        let marking = MarkBoundaries::v47();
        let spaced = marking.marked_stream("e \u{301}x");
        let bare = marking.marked_stream("e\u{301}x");
        assert_ne!(spaced, bare);
        let decoder = Unmark::new();
        assert_eq!(decoder.decode(vec![spaced]).unwrap(), "e \u{301}x");
        assert_eq!(decoder.decode(vec![bare]).unwrap(), "e\u{301}x");
    }
}
