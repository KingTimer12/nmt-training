use burn::{
    nn::loss::CrossEntropyLossConfig,
    tensor::backend::{AutodiffBackend, Backend},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

use crate::{data::batcher::TranslationBatch, model::nmt::NMT, tokenizer::PAD};

impl<B: Backend> NMT<B> {
    pub fn forward_train(&self, batch: TranslationBatch<B>) -> ClassificationOutput<B> {
        let output = self.forward(
            batch.src,
            batch.src_pad_mask,
            Some(batch.tgt_in),
            Some(batch.tgt_pad_mask),
        );

        let [b, t, d] = output.dims();
        let logits = output.reshape([b * t, d]);
        let tgt_out = batch.tgt_out.reshape([b * t]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![PAD as usize]))
            .with_smoothing(Some(0.1))
            .init(&logits.device())
            .forward(logits.clone(), tgt_out.clone());

        ClassificationOutput::new(loss, logits, tgt_out)
    }
}

impl<B: AutodiffBackend> TrainStep for NMT<B> {
    type Input = TranslationBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: TranslationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_train(batch);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for NMT<B> {
    type Input = TranslationBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: TranslationBatch<B>) -> ClassificationOutput<B> {
        self.forward_train(batch)
    }
}
