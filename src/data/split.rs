//! Deterministic train/valid/test split.
//!
//! Every sentence (per language, after normalization) is a node; every pair is an edge.
//! A connected component is one "translation cluster" (e.g. `Run!`, `Run.`, `Corre!`,
//! `Corra!`, `Corram!` all end up together). The whole component gets one split, chosen
//! by hashing its smallest key. This keeps any sentence, on either side, out of two
//! splits, so neither en→pt nor pt→en leaks.

use std::collections::HashMap;

use unicode_normalization::UnicodeNormalization;

use super::record::PairRecord;
use crate::tokenizer::Lang;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    Train,
    Valid,
    Test,
}

#[derive(Debug, Clone, Copy)]
pub struct SplitConfig {
    /// Components with `hash % 1000 < test_per_mille` go to test.
    pub test_per_mille: u64,
    /// The next `valid_per_mille` buckets go to valid; the rest to train.
    pub valid_per_mille: u64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            test_per_mille: 10,
            valid_per_mille: 10,
        }
    }
}

impl SplitConfig {
    pub fn split_for_hash(&self, h: u64) -> Split {
        let bucket = h % 1000;
        if bucket < self.test_per_mille {
            Split::Test
        } else if bucket < self.test_per_mille + self.valid_per_mille {
            Split::Valid
        } else {
            Split::Train
        }
    }
}

/// Key used to decide "same sentence": NFC, lowercase, punctuation/symbols removed,
/// whitespace collapsed. `"Run!"` and `"run."` both become `"run"`.
/// Falls back to the NFC-trimmed text when nothing alphanumeric remains (e.g. `"?!"`).
pub fn normalize_key(text: &str) -> String {
    let nfc: String = text.nfc().collect();
    let mut out = String::with_capacity(nfc.len());
    let mut pending_space = false;
    for c in nfc.chars().flat_map(char::to_lowercase) {
        if c.is_alphanumeric() {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            pending_space = false;
            out.push(c);
        } else {
            pending_space = true;
        }
    }
    if out.is_empty() {
        nfc.trim().to_string()
    } else {
        out
    }
}

/// 64-bit FNV-1a. Hand-rolled because `std`'s `DefaultHasher` is not guaranteed to be
/// stable across Rust versions, and the split must never change.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn node_key(lang: Lang, text: &str) -> String {
    // Language is part of the key: "Tom" (en) and "Tom" (pt_BR) are different nodes.
    format!("{}\u{1f}{}", lang.code(), normalize_key(text))
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

pub struct SplitAssignment {
    /// `splits[i]` is the split of `records[i]`.
    pub splits: Vec<Split>,
    /// `component[i]` is a dense component id of `records[i]` (for stats).
    pub component: Vec<usize>,
    pub num_components: usize,
}

pub fn assign_splits(records: &[PairRecord], cfg: &SplitConfig) -> SplitAssignment {
    let mut ids: HashMap<String, usize> = HashMap::new();
    let mut keys: Vec<String> = Vec::new();
    let mut intern = |k: String| -> usize {
        *ids.entry(k.clone()).or_insert_with(|| {
            keys.push(k);
            keys.len() - 1
        })
    };
    let edges: Vec<(usize, usize)> = records
        .iter()
        .map(|r| {
            (
                intern(node_key(r.src_lang, &r.src)),
                intern(node_key(r.tgt_lang, &r.tgt)),
            )
        })
        .collect();

    let mut uf = UnionFind::new(keys.len());
    for &(a, b) in &edges {
        uf.union(a, b);
    }

    // Representative key of a component = its lexicographically smallest node key, so the
    // result does not depend on record order or union-find internals.
    let mut min_key: HashMap<usize, usize> = HashMap::new();
    for node in 0..keys.len() {
        let root = uf.find(node);
        let e = min_key.entry(root).or_insert(node);
        if keys[node] < keys[*e] {
            *e = node;
        }
    }
    let mut dense: HashMap<usize, usize> = HashMap::new();
    let mut splits = Vec::with_capacity(records.len());
    let mut component = Vec::with_capacity(records.len());
    for &(a, _) in &edges {
        let root = uf.find(a);
        let rep = &keys[min_key[&root]];
        splits.push(cfg.split_for_hash(fnv1a64(rep.as_bytes())));
        let n = dense.len();
        component.push(*dense.entry(root).or_insert(n));
    }
    SplitAssignment {
        splits,
        component,
        num_components: dense.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(src: &str, tgt: &str) -> PairRecord {
        PairRecord {
            id: format!("t:{src}-{tgt}"),
            src_lang: Lang::En,
            tgt_lang: Lang::PtBr,
            src: src.into(),
            tgt: tgt.into(),
            corpus: "t".into(),
            doc: None,
            seg: None,
        }
    }

    #[test]
    fn normalize_key_cases() {
        assert_eq!(normalize_key("Run!"), "run");
        assert_eq!(normalize_key("  Run.  "), "run");
        assert_eq!(normalize_key("I'm   fine,  thanks."), "i m fine thanks");
        assert_eq!(normalize_key("Você está bem?"), "você está bem");
        // NFD input (e + combining circumflex) normalizes to the same key as NFC.
        assert_eq!(normalize_key("Voce\u{302}"), normalize_key("Você"));
        assert_eq!(normalize_key("你好！"), "你好");
        assert_eq!(normalize_key("?!"), "?!");
    }

    #[test]
    fn fnv1a64_known_vectors() {
        // Reference values of the FNV-1a 64-bit spec.
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn split_for_hash_boundaries() {
        let c = SplitConfig::default();
        assert_eq!(c.split_for_hash(0), Split::Test);
        assert_eq!(c.split_for_hash(9), Split::Test);
        assert_eq!(c.split_for_hash(10), Split::Valid);
        assert_eq!(c.split_for_hash(19), Split::Valid);
        assert_eq!(c.split_for_hash(20), Split::Train);
        assert_eq!(c.split_for_hash(1999), Split::Train);
    }

    #[test]
    fn shared_target_links_different_sources() {
        // "Run!" and "Run." normalize equal; "Go." shares the target "Vai." with "Walk."
        // via a chain. The chain Go.–Vai.–Walk.–Ande. must be one component.
        let records = vec![
            rec("Run!", "Corre!"),
            rec("Run.", "Corra!"),
            rec("Go.", "Vai."),
            rec("Walk.", "Vai."),
            rec("Walk.", "Ande."),
            rec("Hello.", "Olá."),
        ];
        let a = assign_splits(&records, &SplitConfig::default());
        assert_eq!(a.component[0], a.component[1]);
        assert_eq!(a.component[2], a.component[3]);
        assert_eq!(a.component[3], a.component[4]);
        assert_ne!(a.component[0], a.component[2]);
        assert_ne!(a.component[2], a.component[5]);
        assert_eq!(a.num_components, 3);
        for (i, j) in [(0, 1), (2, 3), (2, 4)] {
            assert_eq!(a.splits[i], a.splits[j]);
        }
    }

    #[test]
    fn same_text_different_language_not_merged() {
        let mut r1 = rec("Tom.", "Tom.");
        r1.tgt_lang = Lang::PtBr;
        let r2 = rec("Mary.", "Maria.");
        let mut r3 = rec("Maria.", "Mary.");
        // pt_BR "Maria." -> en "Mary." : same nodes as r2, just reversed direction.
        r3.src_lang = Lang::PtBr;
        r3.tgt_lang = Lang::En;
        let a = assign_splits(&[r1, r2, r3], &SplitConfig::default());
        assert_eq!(a.component[1], a.component[2]);
        assert_ne!(a.component[0], a.component[1]);
    }

    #[test]
    fn assignment_independent_of_record_order() {
        let records = vec![
            rec("Run!", "Corre!"),
            rec("Go.", "Vai."),
            rec("Walk.", "Vai."),
            rec("Hello.", "Olá."),
            rec("Hi.", "Oi."),
        ];
        let cfg = SplitConfig {
            test_per_mille: 300,
            valid_per_mille: 300,
        };
        let a = assign_splits(&records, &cfg);
        let mut rev = records.clone();
        rev.reverse();
        let b = assign_splits(&rev, &cfg);
        let n = records.len();
        for i in 0..n {
            assert_eq!(a.splits[i], b.splits[n - 1 - i]);
        }
    }
}
