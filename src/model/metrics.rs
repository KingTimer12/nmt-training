//! Step output and metrics that never copy logits off the device.
//!
//! burn's `ClassificationOutput` syncs the whole `[tokens, vocab]` logits tensor to the CPU
//! every step so that `AccuracyMetric`/`PerplexityMetric` can recompute argmax and
//! log-softmax there. Here the step reduces everything to four scalars on the device and
//! only those are synced.

use burn::{
    backend::NdArray,
    prelude::*,
    train::metric::{
        Adaptor, ItemLazy, LossInput, Metric, MetricAttributes, MetricMetadata, MetricName,
        Numeric, NumericAttributes, NumericEntry, SerializedEntry,
        state::{FormatOptions, NumericMetricState},
    },
};
use std::sync::Arc;

pub struct TranslationOutput<B: Backend> {
    /// Training loss (label-smoothed cross-entropy per non-pad token), `[1]`.
    pub loss: Tensor<B, 1>,
    /// `[loss, nll_sum, correct, tokens]`, detached.
    pub stats: Tensor<B, 1>,
}

/// Synced [`TranslationOutput`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranslationStats {
    pub loss: f64,
    /// Sum over non-pad tokens of `-log p(target)` (no smoothing).
    pub nll_sum: f64,
    /// Non-pad tokens whose argmax equals the target.
    pub correct: f64,
    /// Non-pad target tokens.
    pub tokens: f64,
}

impl<B: Backend> ItemLazy for TranslationOutput<B> {
    type ItemSync = TranslationStats;

    fn sync(self) -> TranslationStats {
        let v: Vec<f32> = self.stats.into_data().convert::<f32>().to_vec().unwrap();
        TranslationStats {
            loss: v[0] as f64,
            nll_sum: v[1] as f64,
            correct: v[2] as f64,
            tokens: v[3] as f64,
        }
    }
}

impl Adaptor<LossInput<NdArray>> for TranslationStats {
    fn adapt(&self) -> LossInput<NdArray> {
        LossInput::new(Tensor::from_floats([self.loss as f32], &Default::default()))
    }
}

impl Adaptor<TranslationStats> for TranslationStats {
    fn adapt(&self) -> TranslationStats {
        *self
    }
}

/// Token-level accuracy over non-pad targets, weighted by token count.
#[derive(Clone)]
pub struct TokenAccuracyMetric {
    name: MetricName,
    state: NumericMetricState,
}

impl TokenAccuracyMetric {
    pub fn new() -> Self {
        Self {
            name: Arc::new("Accuracy".to_string()),
            state: NumericMetricState::new(),
        }
    }
}

impl Metric for TokenAccuracyMetric {
    type Input = TranslationStats;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: Some("%".to_string()),
            higher_is_better: true,
        }
        .into()
    }

    fn update(&mut self, s: &TranslationStats, _metadata: &MetricMetadata) -> SerializedEntry {
        let acc = 100.0 * s.correct / s.tokens.max(1.0);
        self.state.update(
            acc,
            s.tokens as usize,
            FormatOptions::new(self.name()).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl Numeric for TokenAccuracyMetric {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

/// `exp(mean NLL)` over non-pad targets. The running value is the mean NLL of the epoch
/// (summed tokens), exponentiated.
#[derive(Clone)]
pub struct PerplexityMetric {
    name: MetricName,
    /// Tracks mean NLL; perplexity is derived from it.
    nll: NumericMetricState,
}

impl PerplexityMetric {
    pub fn new() -> Self {
        Self {
            name: Arc::new("Perplexity".to_string()),
            nll: NumericMetricState::new(),
        }
    }

    fn exp(entry: NumericEntry) -> NumericEntry {
        match entry {
            NumericEntry::Value(v) => NumericEntry::Value(v.exp()),
            NumericEntry::Aggregated {
                aggregated_value,
                count,
            } => NumericEntry::Aggregated {
                aggregated_value: aggregated_value.exp(),
                count,
            },
        }
    }
}

impl Metric for PerplexityMetric {
    type Input = TranslationStats;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }

    fn update(&mut self, s: &TranslationStats, _metadata: &MetricMetadata) -> SerializedEntry {
        let mean_nll = s.nll_sum / s.tokens.max(1.0);
        self.nll
            .update(mean_nll, s.tokens as usize, FormatOptions::new(self.name()));
        let NumericEntry::Aggregated {
            aggregated_value: epoch_nll,
            ..
        } = self.nll.running_value()
        else {
            unreachable!("NumericMetricState always aggregates")
        };
        let (batch, epoch) = (mean_nll.exp(), epoch_nll.exp());
        let serialized = NumericEntry::Aggregated {
            aggregated_value: batch,
            count: s.tokens as usize,
        }
        .serialize();
        SerializedEntry::new(format!("epoch {epoch:.2} - batch {batch:.2}"), serialized)
    }

    fn clear(&mut self) {
        self.nll.reset()
    }
}

impl Numeric for PerplexityMetric {
    fn value(&self) -> NumericEntry {
        Self::exp(self.nll.current_value())
    }

    fn running_value(&self) -> NumericEntry {
        Self::exp(self.nll.running_value())
    }
}
