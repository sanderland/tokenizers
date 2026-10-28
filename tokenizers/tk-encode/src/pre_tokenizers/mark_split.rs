use crate::pipeline::{self, PreTokenizerScratch};
use crate::tokenizer::Result;

// The boundary marks `MarkBoundaries` emits; see `normalizers::mark_boundaries`.
const BOW: char = '\u{FDD0}';
const EOW: char = '\u{FDD1}';
const SHIFT: char = '\u{FDD3}';
const CAPS: char = '\u{FDD4}';

/// Split a marked stream at its boundary marks: every bow mark starts a
/// pre-token, every eow mark ends one, and whitespace runs stand alone.
/// Case marks ride along with their neighbors. Nothing is dropped and no
/// span is empty, so concatenating the spans recovers the input.
///
/// The trainer feeds these spans (not whole lines) to seeding and EM, so no
/// learned piece ever crosses a boundary mark; inference runs the same split
/// before the lattice.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MarkSplit;

impl MarkSplit {
    /// Byte-offset spans covering `text` in order.
    pub fn split(&self, text: &str) -> Vec<(u32, u32)> {
        let mut spans: Vec<(u32, u32)> = vec![];
        let mut start: u32 = 0;
        let cut = |spans: &mut Vec<(u32, u32)>, at: u32, start: &mut u32| {
            if *start < at {
                spans.push((*start, at));
            }
            *start = at;
        };
        // Cuts at 0 are no-ops through the `start < at` guard. A bow after
        // case marks stays with them: words are spelled `{pre}{bow}…{eow}`,
        // so splitting there would strand pieces like `{shift}{bow}wood`.
        let mut prev_ws = false;
        let mut prev = '\0';
        for (i, c) in text.char_indices() {
            let i = i as u32;
            let ws = c.is_whitespace();
            if (c == BOW && !matches!(prev, SHIFT | CAPS)) || ws != prev_ws {
                cut(&mut spans, i, &mut start);
            }
            prev_ws = ws;
            prev = c;
            if c == EOW {
                cut(&mut spans, i + c.len_utf8() as u32, &mut start);
            }
        }
        cut(&mut spans, text.len() as u32, &mut start);
        spans
    }
}

// SAFETY: every cut falls on a `char_indices` boundary, so all spans are
// valid `&str` ranges of `text`.
unsafe impl pipeline::PreTokenizer for MarkSplit {
    fn pre_tokenize(
        &self,
        text: &str,
        _scratch: &mut PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        for (start, end) in self.split(text) {
            out.push(pipeline::Span { start, end });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOW_S: &str = "\u{FDD0}";
    const EOW_S: &str = "\u{FDD1}";
    const SHIFT_S: &str = "\u{FDD3}";

    fn pieces(text: &str) -> Vec<&str> {
        MarkSplit
            .split(text)
            .iter()
            .map(|&(s, e)| &text[s as usize..e as usize])
            .collect()
    }

    #[test]
    fn test_bow_starts_eow_ends() {
        let text = format!("{BOW_S}hello{EOW_S} {BOW_S}world{EOW_S}");
        assert_eq!(
            pieces(&text),
            vec![
                format!("{BOW_S}hello{EOW_S}"),
                " ".to_string(),
                format!("{BOW_S}world{EOW_S}"),
            ]
        );
    }

    #[test]
    fn test_space_runs_stand_alone() {
        let text = format!("{BOW_S}a{EOW_S}   {BOW_S}b{EOW_S}\n\n{BOW_S}c{EOW_S}");
        assert_eq!(
            pieces(&text),
            vec![
                format!("{BOW_S}a{EOW_S}"),
                "   ".to_string(),
                format!("{BOW_S}b{EOW_S}"),
                "\n\n".to_string(),
                format!("{BOW_S}c{EOW_S}"),
            ]
        );
    }

    #[test]
    fn test_case_marks_ride_along() {
        let text = format!("{BOW_S}a{EOW_S} {SHIFT_S}{BOW_S}b{EOW_S}");
        assert_eq!(
            pieces(&text),
            vec![
                format!("{BOW_S}a{EOW_S}"),
                " ".to_string(),
                format!("{SHIFT_S}{BOW_S}b{EOW_S}"),
            ]
        );
    }

    #[test]
    fn test_adjacent_eow_bow_split() {
        let text = format!("{BOW_S}a{EOW_S}{BOW_S}b{EOW_S}");
        assert_eq!(
            pieces(&text),
            vec![format!("{BOW_S}a{EOW_S}"), format!("{BOW_S}b{EOW_S}"),]
        );
    }

    #[test]
    fn test_split_covers_without_empties() {
        for text in ["", " ", BOW_S, EOW_S, &format!("{EOW_S}{BOW_S}")] {
            let spans = MarkSplit.split(text);
            let mut flat = String::new();
            for (i, &(s, e)) in spans.iter().enumerate() {
                assert!(s < e, "empty span in {text:?}");
                if i > 0 {
                    assert_eq!(s, spans[i - 1].1, "gap in {text:?}");
                }
                flat.push_str(&text[s as usize..e as usize]);
            }
            assert_eq!(flat, text);
        }
    }
}
