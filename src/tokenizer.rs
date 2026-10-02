//! Tokenizer: RWKV World vocabulary (via `rwkv-tokenizer`) plus our special tokens.
//!
//! ID layout (the whole `u16` space, 65 536 IDs):
//! - `0`            : `<|endoftext|>` placeholder from RWKV; the encoder never emits it.
//! - `1..=65529`    : regular RWKV tokens (bytes 0x00..0xFF are IDs 1..=256).
//! - `65530..=65535`: our special tokens (see constants below).
//!
//! Encoding is delegated to the crate (greedy longest-prefix trie). Decoding uses our own
//! byte table because the crate's `decode` panics on IDs >= 65530 and returns `Err` on
//! partial UTF-8, which a model under training produces all the time.

use std::fmt;
use std::fs;
use std::path::Path;

use rwkv_tokenizer::WorldTokenizer;
use serde::{Deserialize, Serialize};

/// Number of IDs used by the RWKV vocabulary file (0..=65529).
pub const RWKV_VOCAB_LEN: usize = 65530;

pub const PAD: u16 = 65530;
pub const BOS: u16 = 65531;
pub const EOS: u16 = 65532;
pub const LANG_EN: u16 = 65533;
pub const LANG_PT_BR: u16 = 65534;
pub const LANG_ZH_HANS: u16 = 65535;

/// Size of the model vocabulary (embedding rows / output logits).
pub const VOCAB_SIZE: usize = 65536;

pub const DEFAULT_VOCAB_PATH: &str = "assets/rwkv_vocab_v20230424.txt";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Lang {
    #[serde(rename = "en")]
    En,
    #[serde(rename = "pt_BR")]
    PtBr,
    #[serde(rename = "zh_Hans")]
    ZhHans,
}

impl Lang {
    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::PtBr => "pt_BR",
            Lang::ZhHans => "zh_Hans",
        }
    }
}

pub fn is_special(id: u16) -> bool {
    id == 0 || id as usize >= RWKV_VOCAB_LEN
}

#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Parse { line: usize, reason: String },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Io(e) => write!(f, "io error reading vocab: {e}"),
            TokenizerError::Parse { line, reason } => {
                write!(f, "vocab parse error at line {line}: {reason}")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        TokenizerError::Io(e)
    }
}

pub struct Tokenizer {
    inner: WorldTokenizer,
    /// `bytes[id]` = raw bytes of token `id`, for `id < RWKV_VOCAB_LEN`.
    bytes: Vec<Vec<u8>>,
}

impl Tokenizer {
    pub fn new(vocab_path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = vocab_path.as_ref();
        let text = fs::read_to_string(path)?;
        let bytes = parse_vocab(&text)?;
        let inner = WorldTokenizer::new(Some(&path.to_string_lossy()))?;
        Ok(Self { inner, bytes })
    }

    /// Text -> regular token IDs (never special tokens).
    pub fn encode(&self, text: &str) -> Vec<u16> {
        self.inner.encode(text)
    }

    /// IDs -> text. Special tokens are skipped; invalid UTF-8 becomes U+FFFD.
    pub fn decode(&self, ids: &[u16]) -> String {
        let mut buf = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if !is_special(id) {
                buf.extend_from_slice(&self.bytes[id as usize]);
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    pub fn token_bytes(&self, id: u16) -> Option<&[u8]> {
        if is_special(id) {
            None
        } else {
            Some(&self.bytes[id as usize])
        }
    }
}

/// Parses `rwkv_vocab_v20230424.txt`. Each line is `<id> <python literal> <byte length>`,
/// where the literal is `'...'`, `"..."` or `b'...'`. The trailing length is used to
/// validate our unescaping.
fn parse_vocab(text: &str) -> Result<Vec<Vec<u8>>, TokenizerError> {
    let mut table = vec![Vec::new(); RWKV_VOCAB_LEN];
    for (i, line) in text.lines().enumerate() {
        let lineno = i + 1;
        let err = |reason: &str| TokenizerError::Parse {
            line: lineno,
            reason: reason.to_string(),
        };
        if line.is_empty() {
            continue;
        }
        let (id, rest) = line.split_once(' ').ok_or_else(|| err("missing id"))?;
        let (literal, len) = rest.rsplit_once(' ').ok_or_else(|| err("missing length"))?;
        let id: usize = id.parse().map_err(|_| err("bad id"))?;
        let len: usize = len.parse().map_err(|_| err("bad length"))?;
        if id == 0 || id >= RWKV_VOCAB_LEN {
            return Err(err("id out of range"));
        }
        let bytes = parse_literal(literal).ok_or_else(|| err("bad literal"))?;
        if bytes.len() != len {
            return Err(err(&format!(
                "length mismatch: parsed {} bytes, file says {len}",
                bytes.len()
            )));
        }
        table[id] = bytes;
    }
    Ok(table)
}

fn parse_literal(lit: &str) -> Option<Vec<u8>> {
    let (is_bytes, lit) = match lit.strip_prefix('b') {
        Some(rest) => (true, rest),
        None => (false, lit),
    };
    let quote = lit.chars().next()?;
    if !(quote == '\'' || quote == '"') || lit.len() < 2 || !lit.ends_with(quote) {
        return None;
    }
    let body = &lit[1..lit.len() - 1];
    let mut out = Vec::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut tmp = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
            continue;
        }
        let esc = chars.next()?;
        let push_char = |out: &mut Vec<u8>, c: char| {
            let mut tmp = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
        };
        match esc {
            '\\' => out.push(b'\\'),
            '\'' => out.push(b'\''),
            '"' => out.push(b'"'),
            'n' => out.push(b'\n'),
            'r' => out.push(b'\r'),
            't' => out.push(b'\t'),
            'x' | 'u' | 'U' => {
                let n = match esc {
                    'x' => 2,
                    'u' => 4,
                    _ => 8,
                };
                let hex: String = chars.by_ref().take(n).collect();
                if hex.len() != n {
                    return None;
                }
                let v = u32::from_str_radix(&hex, 16).ok()?;
                if is_bytes {
                    // In b'...' literals \xNN is a raw byte (only \x appears there).
                    if esc != 'x' {
                        return None;
                    }
                    out.push(v as u8);
                } else {
                    // In str literals \xNN / \uNNNN are code points, encoded as UTF-8.
                    push_char(&mut out, char::from_u32(v)?);
                }
            }
            _ => return None,
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> Tokenizer {
        Tokenizer::new(DEFAULT_VOCAB_PATH).expect("vocab file")
    }

    #[test]
    fn special_ids_fill_u16_space_exactly() {
        let specials = [PAD, BOS, EOS, LANG_EN, LANG_PT_BR, LANG_ZH_HANS];
        for (i, &s) in specials.iter().enumerate() {
            assert_eq!(s as usize, RWKV_VOCAB_LEN + i);
            assert!(is_special(s));
        }
        assert_eq!(LANG_ZH_HANS as usize + 1, VOCAB_SIZE);
    }

    #[test]
    fn byte_table_matches_crate_decode() {
        let t = tok();
        let mut checked = 0;
        for id in 1..RWKV_VOCAB_LEN as u16 {
            if let Ok(s) = t.inner.decode(vec![id]) {
                assert_eq!(s.as_bytes(), t.token_bytes(id).unwrap(), "id {id}");
                checked += 1;
            }
        }
        // Only the few hundred non-UTF-8 byte tokens can't be checked this way.
        assert!(checked > 65000, "checked {checked}");
    }

    #[test]
    fn roundtrip_multilingual() {
        let t = tok();
        for s in [
            "Tom jogava tênis com Maria.",
            "Go.",
            "今天是美好的一天。",
            "a\tb\nc\\'\"",
        ] {
            let ids = t.encode(s);
            assert!(ids.iter().all(|&id| !is_special(id)));
            assert_eq!(t.decode(&ids), s);
        }
    }

    #[test]
    fn decode_skips_specials_and_tolerates_partial_utf8() {
        let t = tok();
        let mut ids = vec![BOS, LANG_PT_BR];
        ids.extend(t.encode("ok"));
        ids.push(EOS);
        ids.push(PAD);
        assert_eq!(t.decode(&ids), "ok");
        // "é" is 0xC3 0xA9; byte tokens are id = byte + 1. Lone 0xC3 is invalid UTF-8.
        assert_eq!(t.decode(&[0xC3 + 1]), "\u{FFFD}");
        assert_eq!(t.decode(&[0xC3 + 1, 0xA9 + 1]), "é");
    }
}
