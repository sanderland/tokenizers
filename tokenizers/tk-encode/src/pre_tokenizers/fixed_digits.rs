//! Fixed ASCII decimal chunks. Non-ASCII numerals retain their existing treatment.
use crate::Result;
use crate::pipeline::{self, PreTokenizerScratch};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedDigits {
    pub group_size: usize,
    pub boundary_markers: bool,
}
impl FixedDigits {
    pub fn single(boundary_markers: bool) -> Self {
        Self {
            group_size: 1,
            boundary_markers,
        }
    }
    pub fn triples() -> Self {
        Self {
            group_size: 3,
            boundary_markers: false,
        }
    }
    pub fn validate(self) -> Result<()> {
        if !matches!(self.group_size, 1 | 3) || (self.group_size == 3 && self.boundary_markers) {
            return Err("FixedDigits supports single digits (optional borders) or RTL triples without borders".into());
        }
        Ok(())
    }
    pub fn inventory(self) -> Vec<String> {
        let mut pieces = Vec::new();
        for width in 1..=self.group_size {
            for value in 0..10usize.pow(width as u32) {
                let digits = format!("{value:0width$}");
                pieces.push(digits.clone());
                if self.boundary_markers {
                    pieces.push(format!("\u{FDD0}{digits}"));
                    pieces.push(format!("{digits}\u{FDD1}"));
                    pieces.push(format!("\u{FDD0}{digits}\u{FDD1}"));
                }
            }
        }
        pieces
    }
}
// SAFETY: ASCII digit endpoints and char_indices endpoints are UTF-8 boundaries.
unsafe impl pipeline::PreTokenizer for FixedDigits {
    fn pre_tokenize(
        &self,
        text: &str,
        _: &mut PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        self.validate()?;
        let bytes = text.as_bytes();
        let mut scan = 0;
        let mut rest = 0;
        while scan < bytes.len() {
            if !bytes[scan].is_ascii_digit() {
                scan += 1;
                continue;
            }
            let start = scan;
            while scan < bytes.len() && bytes[scan].is_ascii_digit() {
                scan += 1;
            }
            let end = scan;
            let bow = self.boundary_markers && text[rest..start].ends_with('\u{FDD0}');
            let eow = self.boundary_markers && text[end..].starts_with('\u{FDD1}');
            let digit_start = start - if bow { 3 } else { 0 };
            if rest < digit_start {
                out.push(pipeline::Span {
                    start: rest as u32,
                    end: digit_start as u32,
                });
            }
            let first = if self.group_size == 3 {
                (end - start - 1) % 3 + 1
            } else {
                1
            };
            let mut a = start;
            let mut width = first;
            while a < end {
                let b = a + width;
                out.push(pipeline::Span {
                    start: (if a == start { digit_start } else { a }) as u32,
                    end: (b + if b == end && eow { 3 } else { 0 }) as u32,
                });
                a = b;
                width = self.group_size;
            }
            scan = end + if eow { 3 } else { 0 };
            rest = scan;
        }
        if rest < text.len() {
            out.push(pipeline::Span {
                start: rest as u32,
                end: text.len() as u32,
            });
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::PreTokenizer;
    fn chunks(policy: FixedDigits, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        policy
            .pre_tokenize(text, &mut PreTokenizerScratch::default(), &mut out)
            .unwrap();
        out.iter().map(|s| text[s.range()].to_string()).collect()
    }
    #[test]
    fn counts_and_leading_zeros() {
        assert_eq!(FixedDigits::single(false).inventory().len(), 10);
        assert_eq!(FixedDigits::single(true).inventory().len(), 40);
        assert_eq!(FixedDigits::triples().inventory().len(), 1110);
        assert_eq!(
            chunks(FixedDigits::triples(), "é1234567.00012"),
            ["é", "1", "234", "567", ".", "00", "012"]
        );
    }
    #[test]
    fn borders_only_on_edge_digits() {
        assert_eq!(
            chunks(FixedDigits::single(true), "\u{FDD0}123\u{FDD1}"),
            ["\u{FDD0}1", "2", "3\u{FDD1}"]
        );
        assert_eq!(
            chunks(FixedDigits::single(true), "\u{FDD0}1\u{FDD1}"),
            ["\u{FDD0}1\u{FDD1}"]
        );
    }
}
