use burn::{
    backend::NdArray,
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        Learner, MetricEarlyStoppingStrategy, StoppingCondition, SupervisedTraining,
        metric::{
            LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};

use crate::{
    data::{batcher::TranslationBatcher, dataset::Corpus},
    model::{
        config::NMTConfig,
        metrics::{PerplexityMetric, TokenAccuracyMetric},
    },
    utils::create_artifact_dir,
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: NMTConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
    /// Padded tokens per batch (`rows * longest sequence`) allowed per item of `batch_size`.
    /// Batches whose sentences are this long or shorter keep all `batch_size` rows; only
    /// longer ones get fewer, so the largest batch stays as big as a typical random one.
    #[config(default = 24)]
    pub tokens_per_item: usize,
    /// Stop when the validation loss has not improved for this many epochs.
    #[config(default = 5)]
    pub early_stopping_patience: usize,
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    corpus: &Corpus,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    corpus
        .vocab
        .save(format!("{artifact_dir}/vocab.json"))
        .expect("Vocabulary should be saved successfully");

    B::seed(&device, config.seed);

    let batcher = TranslationBatcher::default();

    let max_tokens = config.batch_size * config.tokens_per_item;
    // Datasets yield whole length-bucketed batches, hence `batch_size(1)`.
    let dataloader_train = DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(
            corpus
                .train
                .bucketed(config.batch_size, max_tokens, config.seed),
        );

    // Validation order does not affect the metrics, so no shuffling.
    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .num_workers(config.num_workers)
        .build(
            corpus
                .valid
                .bucketed(config.batch_size, max_tokens, config.seed),
        );

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((
            TokenAccuracyMetric::new(),
            LossMetric::<NdArray>::new(),
            PerplexityMetric::new(),
        ))
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &LossMetric::<B>::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince {
                n_epochs: config.early_stopping_patience,
            },
        ))
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<B>(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
