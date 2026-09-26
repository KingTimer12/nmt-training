use burn::{
    Tensor,
    tensor::{Int, TensorData, backend::Backend},
};

use crate::{data::{batcher::PaddedBatch, dataset::TranslationItem}, tokenizer::{BOS, EOS, PAD}};

pub fn int2<B: Backend>(
    v: Vec<i64>,
    v_len: usize,
    batch_size: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::<B, 2, Int>::from_data(TensorData::new(v, [batch_size, v_len]), device)
}

pub fn pad_items(items: &[TranslationItem]) -> PaddedBatch {
    assert!(!items.is_empty(), "cannot batch zero items");
    let batch_size = items.len();
    let src_len = items.iter().map(|it| it.src_model_len()).max().unwrap();
    let tgt_len = items.iter().map(|it| it.tgt_model_len()).max().unwrap();
    let pad = PAD as i64;

    let mut src = Vec::with_capacity(batch_size * src_len);
    let mut tgt_in = Vec::with_capacity(batch_size * tgt_len);
    let mut tgt_out = Vec::with_capacity(batch_size * tgt_len);
    for it in items {
        let row_start = src.len();
        src.push(it.tgt_lang.tag() as i64);
        src.extend(it.src.iter().map(|&id| id as i64));
        src.push(EOS as i64);
        src.resize(row_start + src_len, pad);

        let row_start = tgt_in.len();
        tgt_in.push(BOS as i64);
        tgt_in.extend(it.tgt.iter().map(|&id| id as i64));
        tgt_in.resize(row_start + tgt_len, pad);

        let row_start = tgt_out.len();
        tgt_out.extend(it.tgt.iter().map(|&id| id as i64));
        tgt_out.push(EOS as i64);
        tgt_out.resize(row_start + tgt_len, pad);
    }
    PaddedBatch {
        batch_size,
        src_len,
        tgt_len,
        src,
        tgt_in,
        tgt_out,
    }
}

pub fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}