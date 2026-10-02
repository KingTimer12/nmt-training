use burn::{
    prelude::*,
    tensor::{
        activation::log_softmax,
        backend::{AutodiffBackend, Backend},
    },
    train::{InferenceStep, TrainOutput, TrainStep},
};

use crate::{
    data::{batcher::TranslationBatch, vocab::PAD},
    model::{metrics::TranslationOutput, nmt::NMT},
};

const LABEL_SMOOTHING: f32 = 0.1;

impl<B: Backend> NMT<B> {
    pub fn forward_train(&self, batch: TranslationBatch<B>) -> TranslationOutput<B> {
        let output = self.forward(
            batch.src,
            batch.src_pad_mask,
            Some(batch.tgt_in),
            Some(batch.tgt_pad_mask),
        );

        let [b, t, v] = output.dims();
        translation_loss(output.reshape([b * t, v]), batch.tgt_out.reshape([b * t]))
    }
}

/// Loss and metric stats for `logits` `[n, V]` against `targets` `[n]` (`PAD` = ignored).
pub fn translation_loss<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
) -> TranslationOutput<B> {
    let [n, _] = logits.dims();
    let valid = targets.clone().not_equal_elem(PAD as i64).float();

    // Label-smoothed cross-entropy against y·(1-α) + α/V, written as
    // -(1-α)·log p(target) - α·mean(log p). Same value as burn's smoothed loss, without
    // materializing the [n, V] one-hot target matrix and its pad mask.
    let log_probs = log_softmax(logits.clone(), 1);
    let target_lp = log_probs
        .clone()
        .gather(1, targets.clone().reshape([n, 1]))
        .reshape([n]);
    let mean_lp = log_probs.mean_dim(1).reshape([n]);
    let per_token = (target_lp.clone() * (1.0 - LABEL_SMOOTHING) + mean_lp * LABEL_SMOOTHING).neg();
    let tokens = valid.clone().sum();
    // Averaged over real tokens, not over the padded `b * t`.
    let loss = (per_token * valid.clone()).sum() / tokens.clone().clamp_min(1.0);

    let correct = (logits
        .detach()
        .argmax(1)
        .reshape([n])
        .equal(targets)
        .float()
        * valid.clone())
    .sum();
    let nll_sum = (target_lp.detach() * valid).sum().neg();
    let stats = Tensor::cat(vec![loss.clone().detach(), nll_sum, correct, tokens], 0);

    TranslationOutput { loss, stats }
}

impl<B: AutodiffBackend> TrainStep for NMT<B> {
    type Input = TranslationBatch<B>;
    type Output = TranslationOutput<B>;

    fn step(&self, batch: TranslationBatch<B>) -> TrainOutput<TranslationOutput<B>> {
        let item = self.forward_train(batch);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for NMT<B> {
    type Input = TranslationBatch<B>;
    type Output = TranslationOutput<B>;

    fn step(&self, batch: TranslationBatch<B>) -> TranslationOutput<B> {
        self.forward_train(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::metrics::TranslationStats;
    use burn::{backend::NdArray, nn::loss::CrossEntropyLossConfig, train::metric::ItemLazy};

    type TB = NdArray;

    #[test]
    fn matches_burn_smoothed_cross_entropy() {
        let device = Default::default();
        let (n, v) = (7, 11);
        TB::seed(&device, 3);
        let logits =
            Tensor::<TB, 2>::random([n, v], burn::tensor::Distribution::Normal(0., 2.), &device);
        let targets =
            Tensor::<TB, 1, Int>::from_ints([3, PAD as i32, 10, 1, PAD as i32, 7, 2], &device);

        let reference = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![PAD as usize]))
            .with_smoothing(Some(LABEL_SMOOTHING))
            .init(&device)
            .forward(logits.clone(), targets.clone())
            .into_scalar();
        let out = translation_loss(logits.clone(), targets.clone());
        let ours = out.loss.clone().into_scalar();
        // burn divides by all `n` rows, we divide by the 5 non-pad ones.
        assert!(
            (ours * 5.0 / n as f32 - reference).abs() < 1e-5,
            "{ours} vs {reference}"
        );
        let s: TranslationStats = out.sync();

        let plain = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![PAD as usize]))
            .init(&device)
            .forward(logits.clone(), targets.clone())
            .into_scalar();
        let correct: i64 = logits
            .argmax(1)
            .reshape([n])
            .equal(targets.clone())
            .bool_and(targets.not_equal_elem(PAD as i64))
            .int()
            .sum()
            .into_scalar();
        assert_eq!(s.tokens, 5.0);
        assert_eq!(s.correct, correct as f64);
        assert!((s.loss - ours as f64).abs() < 1e-6);
        assert!(
            (s.nll_sum - plain as f64 * n as f64).abs() < 1e-4,
            "{} vs {}",
            s.nll_sum,
            plain * n as f32
        );
    }
}
