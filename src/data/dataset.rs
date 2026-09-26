//! `burn` Dataset of pre-tokenized translation examples.
//!
//! Each canonical pair yields two items (A→B and B→A). Items hold token IDs only;
//! special tokens (`<2xx>`, `<bos>`, `<eos>`, `<pad>`) are added by the batcher.

use std::collections::HashMap;

use burn::data::dataset::Dataset;

use super::record::PairRecord;
use super::split::{Split, SplitAssignment};
use crate::CORPUS;
use crate::data::record::load_jsonl;
use crate::data::split::{SplitConfig, assign_splits};
use crate::tokenizer::{DEFAULT_VOCAB_PATH, Lang, Tokenizer};

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationItem {
    pub src_lang: Lang,
    pub tgt_lang: Lang,
    /// Regular token IDs of the source text, no specials.
    pub src: Vec<u16>,
    /// Regular token IDs of the target text, no specials.
    pub tgt: Vec<u16>,
}

impl TranslationItem {
    /// Length of `src` as the model sees it: `<2xx> src <eos>`.
    pub fn src_model_len(&self) -> usize {
        self.src.len() + 2
    }

    /// Length of `tgt_in` / `tgt_out`: `<bos> tgt` / `tgt <eos>`.
    pub fn tgt_model_len(&self) -> usize {
        self.tgt.len() + 1
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DatasetOptions {
    /// Drop items whose model-side src or tgt length would exceed this.
    pub max_len: usize,
    /// Emit both directions of each pair.
    pub both_directions: bool,
}

impl Default for DatasetOptions {
    fn default() -> Self {
        Self {
            max_len: 128,
            both_directions: true,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BuildStats {
    pub pairs_in_split: usize,
    pub items: usize,
    pub dropped_too_long: usize,
    pub dropped_empty: usize,
}

pub struct TranslationDataset {
    items: Vec<TranslationItem>,
}

impl TranslationDataset {
    pub fn train() -> Self {
        Self::new(Split::Train).0
    }

    pub fn test() -> Self {
        Self::new(Split::Valid).0
    }

    fn new(split: Split) -> (Self, BuildStats) {
        let tokenizer = Tokenizer::new(DEFAULT_VOCAB_PATH).unwrap();
        let records = load_jsonl(CORPUS).unwrap();
        let assignment = assign_splits(&records, &SplitConfig::default());
        println!(
            "{} pairs, {} components",
            records.len(),
            assignment.num_components
        );

        let mut comp_sizes: HashMap<usize, usize> = HashMap::new();
        for &c in &assignment.component {
            *comp_sizes.entry(c).or_default() += 1;
        }
        let mut sizes: Vec<usize> = comp_sizes.values().copied().collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let opts = DatasetOptions::default();

        Self::build(&records, &assignment, split, &tokenizer, &opts)
    }

    pub fn from_items(items: Vec<TranslationItem>) -> Self {
        Self { items }
    }

    fn build(
        records: &[PairRecord],
        assignment: &SplitAssignment,
        split: Split,
        tokenizer: &Tokenizer,
        opts: &DatasetOptions,
    ) -> (Self, BuildStats) {
        assert_eq!(records.len(), assignment.splits.len());
        let mut stats = BuildStats::default();
        let mut items = Vec::new();
        for (rec, _) in records
            .iter()
            .zip(&assignment.splits)
            .filter(|&(_, s)| *s == split)
        {
            stats.pairs_in_split += 1;
            let a = tokenizer.encode(&rec.src);
            let b = tokenizer.encode(&rec.tgt);
            let forward = TranslationItem {
                src_lang: rec.src_lang,
                tgt_lang: rec.tgt_lang,
                src: a.clone(),
                tgt: b.clone(),
            };
            let mut candidates = vec![forward];
            if opts.both_directions {
                candidates.push(TranslationItem {
                    src_lang: rec.tgt_lang,
                    tgt_lang: rec.src_lang,
                    src: b,
                    tgt: a,
                });
            }
            for item in candidates {
                if item.src.is_empty() || item.tgt.is_empty() {
                    stats.dropped_empty += 1;
                } else if item.src_model_len() > opts.max_len || item.tgt_model_len() > opts.max_len
                {
                    stats.dropped_too_long += 1;
                } else {
                    items.push(item);
                }
            }
        }
        stats.items = items.len();
        (Self { items }, stats)
    }
}

impl Dataset<TranslationItem> for TranslationDataset {
    fn get(&self, index: usize) -> Option<TranslationItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::split::{SplitConfig, assign_splits};
    use crate::tokenizer::DEFAULT_VOCAB_PATH;

    fn rec(src: &str, tgt: &str) -> PairRecord {
        PairRecord {
            id: "t".into(),
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
    fn both_directions_and_filtering() {
        let tok = Tokenizer::new(DEFAULT_VOCAB_PATH).unwrap();
        let records = vec![
            rec("Go.", "Vai."),
            rec("Hi.", ""),
            rec(&"x ".repeat(50), "y"),
        ];
        // Everything in train.
        let cfg = SplitConfig {
            test_per_mille: 0,
            valid_per_mille: 0,
        };
        let a = assign_splits(&records, &cfg);
        let opts = DatasetOptions {
            max_len: 20,
            both_directions: true,
        };
        let (ds, stats) = TranslationDataset::build(&records, &a, Split::Train, &tok, &opts);
        assert_eq!(stats.pairs_in_split, 3);
        assert_eq!(stats.dropped_empty, 2);
        // "x x x ..." is dozens of tokens: too long as src (fwd) and as tgt (rev).
        assert_eq!(stats.dropped_too_long, 2);
        assert_eq!(ds.len(), 2);

        let fwd = ds.get(0).unwrap();
        let rev = ds.get(1).unwrap();
        assert_eq!((fwd.src_lang, fwd.tgt_lang), (Lang::En, Lang::PtBr));
        assert_eq!((rev.src_lang, rev.tgt_lang), (Lang::PtBr, Lang::En));
        assert_eq!(fwd.src, tok.encode("Go."));
        assert_eq!(fwd.tgt, tok.encode("Vai."));
        assert_eq!(rev.src, fwd.tgt);
        assert_eq!(rev.tgt, fwd.src);
        assert!(ds.get(2).is_none());
    }

    #[test]
    fn only_requested_split() {
        let tok = Tokenizer::new(DEFAULT_VOCAB_PATH).unwrap();
        let records = vec![rec("Go.", "Vai.")];
        let a = assign_splits(&records, &SplitConfig::default());
        let other = match a.splits[0] {
            Split::Train => Split::Test,
            _ => Split::Train,
        };
        let (ds, _) =
            TranslationDataset::build(&records, &a, other, &tok, &DatasetOptions::default());
        assert!(ds.is_empty());
    }
}
