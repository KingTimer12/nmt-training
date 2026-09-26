//! Batcher: pads items and builds encoder/decoder tensors.
//!
//! For an item `src_lang→tgt_lang` with token IDs `s` and `t`:
//! ```text
//! src     = <2tgt_lang> s... <eos> <pad>...
//! tgt_in  = <bos>       t...       <pad>...   (decoder input, teacher forcing)
//! tgt_out = t...        <eos>      <pad>...   (labels = tgt_in shifted left by one)
//! ```
//! Masks follow burn's convention: `true` = padding = ignored by attention.
//! The causal (autoregressive) mask is not built here; the decoder creates it.

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

use super::dataset::TranslationItem;
use crate::{tokenizer::PAD, utils};

#[derive(Debug, Clone)]
pub struct TranslationBatch<B: Backend> {
    /// `[batch, src_len]`
    pub src: Tensor<B, 2, Int>,
    /// `[batch, src_len]`, `true` where `src` is `<pad>`.
    pub src_pad_mask: Tensor<B, 2, Bool>,
    /// `[batch, tgt_len]`
    pub tgt_in: Tensor<B, 2, Int>,
    /// `[batch, tgt_len]`
    pub tgt_out: Tensor<B, 2, Int>,
    /// `[batch, tgt_len]`, `true` where `tgt_in` is `<pad>`. Same positions are `<pad>`
    /// in `tgt_out`, so it also masks the loss.
    pub tgt_pad_mask: Tensor<B, 2, Bool>,
}

/// Padded rows as plain vectors, row-major. Separated from tensor creation so the
/// layout logic is testable without a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaddedBatch {
    pub batch_size: usize,
    pub src_len: usize,
    pub tgt_len: usize,
    pub src: Vec<i64>,
    pub tgt_in: Vec<i64>,
    pub tgt_out: Vec<i64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TranslationBatcher;

impl<B: Backend> Batcher<B, TranslationItem, TranslationBatch<B>> for TranslationBatcher {
    fn batch(&self, items: Vec<TranslationItem>, device: &B::Device) -> TranslationBatch<B> {
        let p = utils::pad_items(&items);
        let src = utils::int2(p.src, p.src_len, p.batch_size, device);
        let tgt_in = utils::int2(p.tgt_in, p.tgt_len, p.batch_size, device);
        let tgt_out = utils::int2(p.tgt_out, p.tgt_len, p.batch_size, device);
        // Regular tokens are always < PAD, so equality with PAD identifies padding exactly.
        let src_pad_mask = src.clone().equal_elem(PAD as i64);
        let tgt_pad_mask = tgt_in.clone().equal_elem(PAD as i64);
        TranslationBatch {
            src,
            src_pad_mask,
            tgt_in,
            tgt_out,
            tgt_pad_mask,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{BOS, EOS, LANG_EN, LANG_PT_BR, Lang};
    use burn::backend::NdArray;

    type B = NdArray;

    const P: i64 = PAD as i64;
    const BO: i64 = BOS as i64;
    const EO: i64 = EOS as i64;

    fn item(src: &[u16], tgt: &[u16], tgt_lang: Lang) -> TranslationItem {
        TranslationItem {
            src_lang: if tgt_lang == Lang::En {
                Lang::PtBr
            } else {
                Lang::En
            },
            tgt_lang,
            src: src.to_vec(),
            tgt: tgt.to_vec(),
        }
    }

    fn two_items() -> Vec<TranslationItem> {
        vec![
            item(&[10, 11, 12], &[20], Lang::PtBr),
            item(&[13], &[21, 22, 23, 24], Lang::En),
        ]
    }

    #[test]
    fn pad_items_layout() {
        let p = utils::pad_items(&two_items());
        assert_eq!((p.batch_size, p.src_len, p.tgt_len), (2, 5, 5));
        let tp = LANG_PT_BR as i64;
        let te = LANG_EN as i64;
        #[rustfmt::skip]
        assert_eq!(p.src, vec![
            tp, 10, 11, 12, EO,
            te, 13, EO, P,  P,
        ]);
        #[rustfmt::skip]
        assert_eq!(p.tgt_in, vec![
            BO, 20, P,  P,  P,
            BO, 21, 22, 23, 24,
        ]);
        #[rustfmt::skip]
        assert_eq!(p.tgt_out, vec![
            20, EO, P,  P,  P,
            21, 22, 23, 24, EO,
        ]);
    }

    #[test]
    fn tgt_out_is_tgt_in_shifted_left() {
        let p = utils::pad_items(&two_items());
        for r in 0..p.batch_size {
            let row = |v: &[i64]| v[r * p.tgt_len..(r + 1) * p.tgt_len].to_vec();
            let (tin, tout) = (row(&p.tgt_in), row(&p.tgt_out));
            assert_eq!(tin[0], BO);
            // tgt_in[1..] == tgt_out[..n-1] wherever tgt_in is not padding.
            for i in 1..p.tgt_len {
                if tin[i] != P {
                    assert_eq!(tin[i], tout[i - 1]);
                }
            }
            // Exactly one <eos> per row, and pad positions coincide in tgt_in/tgt_out.
            assert_eq!(tout.iter().filter(|&&x| x == EO).count(), 1);
            for i in 0..p.tgt_len {
                assert_eq!(tin[i] == P, tout[i] == P, "row {r} pos {i}");
            }
        }
    }

    #[test]
    fn batch_tensor_shapes_and_masks() {
        let device = Default::default();
        let b: TranslationBatch<B> = TranslationBatcher.batch(two_items(), &device);
        assert_eq!(b.src.dims(), [2, 5]);
        assert_eq!(b.src_pad_mask.dims(), [2, 5]);
        assert_eq!(b.tgt_in.dims(), [2, 5]);
        assert_eq!(b.tgt_out.dims(), [2, 5]);
        assert_eq!(b.tgt_pad_mask.dims(), [2, 5]);

        let src_mask: Vec<bool> = b.src_pad_mask.into_data().to_vec().unwrap();
        #[rustfmt::skip]
        assert_eq!(src_mask, vec![
            false, false, false, false, false,
            false, false, false, true,  true,
        ]);
        let tgt_mask: Vec<bool> = b.tgt_pad_mask.into_data().to_vec().unwrap();
        #[rustfmt::skip]
        assert_eq!(tgt_mask, vec![
            false, false, true,  true,  true,
            false, false, false, false, false,
        ]);
        let src: Vec<i64> = b.src.into_data().convert::<i64>().to_vec().unwrap();
        assert_eq!(src, utils::pad_items(&two_items()).src);
    }

    #[test]
    fn single_item_has_no_padding() {
        let p = utils::pad_items(&[item(&[5, 6], &[7, 8, 9], Lang::PtBr)]);
        assert_eq!((p.src_len, p.tgt_len), (4, 4));
        assert!(!p.src.contains(&P) && !p.tgt_in.contains(&P) && !p.tgt_out.contains(&P));
    }

    #[test]
    #[should_panic(expected = "zero items")]
    fn empty_batch_panics() {
        utils::pad_items(&[]);
    }
}
