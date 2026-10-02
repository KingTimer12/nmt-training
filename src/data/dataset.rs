//! `burn` Dataset of pre-tokenized translation examples.
//!
//! Each canonical pair yields two items (A→B and B→A). Items hold token IDs only;
//! special tokens (`<2xx>`, `<bos>`, `<eos>`, `<pad>`) are added by the batcher.

use std::sync::Arc;

use burn::data::dataset::Dataset;

use super::record::PairRecord;
use super::split::{Split, SplitAssignment};
use super::vocab::Vocab;
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
    items: Arc<Vec<TranslationItem>>,
}

/// Train and validation datasets plus the compact vocabulary they are encoded with.
pub struct Corpus {
    pub train: TranslationDataset,
    pub valid: TranslationDataset,
    pub vocab: Vocab,
}

impl Corpus {
    /// Loads the corpus once: one JSONL parse, one split assignment and one tokenization
    /// per record shared by both splits.
    pub fn load() -> Self {
        let tokenizer = Tokenizer::new(DEFAULT_VOCAB_PATH).unwrap();
        let records = load_jsonl(CORPUS).unwrap();
        let assignment = assign_splits(&records, &SplitConfig::default());
        println!(
            "{} pairs, {} components",
            records.len(),
            assignment.num_components
        );

        let tokens = tokenize(&records, &tokenizer);
        // Built from every split so valid/test never meet an unknown token; this only
        // decides which embedding rows exist, not what is trained on.
        let vocab = Vocab::from_tokens(tokens.iter().flat_map(|(a, b)| a.iter().chain(b)).copied());
        let tokens: Vec<_> = tokens
            .iter()
            .map(|(a, b)| (vocab.encode(a), vocab.encode(b)))
            .collect();
        println!("vocabulary: {} tokens", vocab.len());

        let opts = DatasetOptions::default();
        let build = |split| {
            let (ds, stats) =
                TranslationDataset::build(&records, &tokens, &assignment, split, &opts);
            println!("{split:?}: {stats:?}");
            ds
        };
        Self {
            train: build(Split::Train),
            valid: build(Split::Valid),
            vocab,
        }
    }
}

/// Encodes `(src, tgt)` of every record.
pub fn tokenize(records: &[PairRecord], tokenizer: &Tokenizer) -> Vec<(Vec<u16>, Vec<u16>)> {
    records
        .iter()
        .map(|r| (tokenizer.encode(&r.src), tokenizer.encode(&r.tgt)))
        .collect()
}

impl TranslationDataset {
    pub fn from_items(items: Vec<TranslationItem>) -> Self {
        Self {
            items: Arc::new(items),
        }
    }

    /// `tokens[i]` is the encoded `(src, tgt)` of `records[i]`.
    fn build(
        records: &[PairRecord],
        tokens: &[(Vec<u16>, Vec<u16>)],
        assignment: &SplitAssignment,
        split: Split,
        opts: &DatasetOptions,
    ) -> (Self, BuildStats) {
        assert_eq!(records.len(), assignment.splits.len());
        assert_eq!(records.len(), tokens.len());
        let mut stats = BuildStats::default();
        let mut items = Vec::new();
        for ((rec, (a, b)), _) in records
            .iter()
            .zip(tokens)
            .zip(&assignment.splits)
            .filter(|&(_, s)| *s == split)
        {
            stats.pairs_in_split += 1;
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
                    src: b.clone(),
                    tgt: a.clone(),
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
        (Self::from_items(items), stats)
    }

    /// Groups items of similar length into batches of at most `batch_size` items, so a batch
    /// is padded to barely more than its own lengths instead of the longest of a random sample.
    ///
    /// Items are sorted by `(src_len, tgt_len)` with a seeded random tie-break, then cut into
    /// consecutive chunks. A chunk also ends before `rows * longest_len` would exceed
    /// `max_tokens`: sorting puts all the longest sentences in the same batch, which would
    /// otherwise be far bigger than any random batch. Batch contents are fixed; the
    /// dataloader shuffles batch order every epoch (same scheme as fairseq).
    pub fn bucketed(&self, batch_size: usize, max_tokens: usize, seed: u64) -> BucketedDataset {
        assert!(batch_size > 0);
        let len = |i: u32| {
            let it = &self.items[i as usize];
            (it.src_model_len(), it.tgt_model_len())
        };
        let mut order: Vec<u32> = (0..self.items.len() as u32).collect();
        order.sort_by_cached_key(|&i| (len(i), splitmix64(seed ^ i as u64)));

        let mut batches = Vec::new();
        let mut batch: Vec<u32> = Vec::new();
        let mut longest = 0;
        for i in order {
            let (s, t) = len(i);
            let l = longest.max(s).max(t);
            if !batch.is_empty()
                && (batch.len() == batch_size || (batch.len() + 1) * l > max_tokens)
            {
                batches.push(std::mem::take(&mut batch));
                longest = 0;
            }
            longest = longest.max(s).max(t);
            batch.push(i);
        }
        if !batch.is_empty() {
            batches.push(batch);
        }
        BucketedDataset {
            items: self.items.clone(),
            batches,
        }
    }
}

/// Stateless 64-bit mixer (SplitMix64 finalizer), used as a seeded per-item random key.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl Dataset<TranslationItem> for TranslationDataset {
    fn get(&self, index: usize) -> Option<TranslationItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// A dataset whose items are whole, length-bucketed batches. Use it with a dataloader
/// `batch_size` of 1.
pub struct BucketedDataset {
    items: Arc<Vec<TranslationItem>>,
    batches: Vec<Vec<u32>>,
}

impl Dataset<Vec<TranslationItem>> for BucketedDataset {
    fn get(&self, index: usize) -> Option<Vec<TranslationItem>> {
        let batch = self.batches.get(index)?;
        Some(
            batch
                .iter()
                .map(|&i| self.items[i as usize].clone())
                .collect(),
        )
    }

    fn len(&self) -> usize {
        self.batches.len()
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
        let tokens = tokenize(&records, &tok);
        let (ds, stats) = TranslationDataset::build(&records, &tokens, &a, Split::Train, &opts);
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
        let tokens = tokenize(&records, &tok);
        let (ds, _) =
            TranslationDataset::build(&records, &tokens, &a, other, &DatasetOptions::default());
        assert!(ds.is_empty());
    }

    #[test]
    fn bucketed_batches_group_similar_lengths() {
        let item = |n: usize| TranslationItem {
            src_lang: Lang::En,
            tgt_lang: Lang::PtBr,
            src: vec![7; n],
            tgt: vec![8; n],
        };
        let lens = [5, 1, 9, 1, 5, 9, 3];
        let ds = TranslationDataset::from_items(lens.iter().map(|&n| item(n)).collect());
        let b = ds.bucketed(2, usize::MAX, 42);
        assert_eq!(b.len(), 4);
        let got: Vec<Vec<usize>> = (0..b.len())
            .map(|i| b.get(i).unwrap().iter().map(|it| it.src.len()).collect())
            .collect();
        assert_eq!(got, vec![vec![1, 1], vec![3, 5], vec![5, 9], vec![9]]);
        assert!(b.get(4).is_none());
    }

    #[test]
    fn bucketed_respects_token_budget() {
        let item = |n: usize| TranslationItem {
            src_lang: Lang::En,
            tgt_lang: Lang::PtBr,
            src: vec![7; n],
            tgt: vec![8; 1],
        };
        // Model lengths (src + 2): 3, 3, 3, 12, 12.
        let ds = TranslationDataset::from_items([1, 1, 1, 10, 10].map(item).to_vec());
        let b = ds.bucketed(4, 20, 0);
        let sizes: Vec<usize> = (0..b.len()).map(|i| b.get(i).unwrap().len()).collect();
        // 3 short rows fit (3 * 3 <= 20); a 12-long row then needs its own batch each.
        assert_eq!(sizes, vec![3, 1, 1]);
    }
}
