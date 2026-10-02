//! Compact model vocabulary.
//!
//! The RWKV vocabulary has 65 536 IDs, but a corpus only uses a fraction of them. Every
//! unused ID still costs an embedding row, an output logit per target token, its share of
//! the softmax and Adam state, so the model works on a dense ID space instead:
//! - `0..NUM_SPECIAL`: our special tokens (constants below);
//! - `NUM_SPECIAL..`  : the RWKV tokens that occur in the corpus, in RWKV ID order.
//!
//! Tokens that never occur only ever receive "push down" gradients, so dropping them does
//! not change what the model learns about the corpus.

use std::io;
use std::path::Path;

use crate::tokenizer::{self, Lang, VOCAB_SIZE, is_special};

pub const PAD: u16 = 0;
pub const BOS: u16 = 1;
pub const EOS: u16 = 2;
pub const LANG_EN: u16 = 3;
pub const LANG_PT_BR: u16 = 4;
pub const LANG_ZH_HANS: u16 = 5;
pub const NUM_SPECIAL: usize = 6;

/// RWKV-space IDs of the specials, indexed by their compact ID.
const SPECIALS_RWKV: [u16; NUM_SPECIAL] = [
    tokenizer::PAD,
    tokenizer::BOS,
    tokenizer::EOS,
    tokenizer::LANG_EN,
    tokenizer::LANG_PT_BR,
    tokenizer::LANG_ZH_HANS,
];

const ABSENT: u16 = u16::MAX;

/// The `<2xx>` token prepended to the source to select the target language.
pub fn lang_tag(lang: Lang) -> u16 {
    match lang {
        Lang::En => LANG_EN,
        Lang::PtBr => LANG_PT_BR,
        Lang::ZhHans => LANG_ZH_HANS,
    }
}

#[derive(Debug, Clone)]
pub struct Vocab {
    /// `to_rwkv[compact]` = RWKV ID.
    to_rwkv: Vec<u16>,
    /// `to_compact[rwkv]` = compact ID, or `ABSENT`.
    to_compact: Vec<u16>,
}

impl Vocab {
    /// Builds the vocabulary from every regular RWKV token that occurs in `tokens`.
    pub fn from_tokens(tokens: impl IntoIterator<Item = u16>) -> Self {
        let mut seen = vec![false; VOCAB_SIZE];
        for id in tokens {
            assert!(!is_special(id), "special token {id} in tokenized text");
            seen[id as usize] = true;
        }
        let mut to_rwkv = SPECIALS_RWKV.to_vec();
        to_rwkv.extend((0..VOCAB_SIZE).filter(|&id| seen[id]).map(|id| id as u16));

        let mut to_compact = vec![ABSENT; VOCAB_SIZE];
        for (compact, &rwkv) in to_rwkv.iter().enumerate() {
            to_compact[rwkv as usize] = compact as u16;
        }
        Self {
            to_rwkv,
            to_compact,
        }
    }

    /// Number of compact IDs (embedding rows / output logits).
    pub fn len(&self) -> usize {
        self.to_rwkv.len()
    }

    /// Regular RWKV IDs -> compact IDs. Panics on a token not in the vocabulary.
    pub fn encode(&self, rwkv_ids: &[u16]) -> Vec<u16> {
        rwkv_ids
            .iter()
            .map(|&id| {
                let c = self.to_compact[id as usize];
                assert!(c != ABSENT, "token {id} not in vocabulary");
                c
            })
            .collect()
    }

    /// Compact IDs -> RWKV IDs, ready for `Tokenizer::decode`.
    pub fn decode(&self, ids: &[u16]) -> Vec<u16> {
        ids.iter().map(|&id| self.to_rwkv[id as usize]).collect()
    }

    /// Saves the compact -> RWKV table as a JSON array, needed to decode a trained model.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, serde_json::to_string(&self.to_rwkv)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specials_first_then_sorted_regular() {
        let v = Vocab::from_tokens([300, 7, 300, 42]);
        assert_eq!(v.len(), NUM_SPECIAL + 3);
        assert_eq!(v.encode(&[7, 42, 300]), vec![6, 7, 8]);
        assert_eq!(
            v.decode(&[PAD, EOS, 6, 8]),
            vec![tokenizer::PAD, tokenizer::EOS, 7, 300]
        );
        assert_eq!(lang_tag(Lang::PtBr), LANG_PT_BR);
        assert_eq!(v.decode(&[LANG_PT_BR]), vec![tokenizer::LANG_PT_BR]);
    }

    #[test]
    #[should_panic(expected = "not in vocabulary")]
    fn unknown_token_panics() {
        Vocab::from_tokens([7]).encode(&[8]);
    }
}
