use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        Learner, MetricEarlyStoppingStrategy, StoppingCondition, SupervisedTraining,
        metric::{
            AccuracyMetric, LossMetric, PerplexityMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};

use crate::{
    data::{batcher::TranslationBatcher, dataset::TranslationDataset},
    model::config::NMTConfig,
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
    /// Stop when the validation loss has not improved for this many epochs.
    #[config(default = 5)]
    pub early_stopping_patience: usize,
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let batcher = TranslationBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TranslationDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TranslationDataset::test());

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((
            AccuracyMetric::new(),
            LossMetric::new(),
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
