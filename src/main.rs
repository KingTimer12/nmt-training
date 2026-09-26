use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;

use crate::model::config::NMTConfig;
use crate::tokenizer::VOCAB_SIZE;
use crate::training::{TrainingConfig, train};

mod data;
pub mod model;
mod tokenizer;
mod training;
pub mod utils;

pub const CORPUS: &str = "data/canonical/en-pt_BR/tatoeba.jsonl";

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

const D_MODEL: usize = 256; // quanto maior, mais pesado e lento será o treinamento, mas melhor será a qualidade da tradução
const N_EMBEDDINGS: usize = VOCAB_SIZE;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::DiscreteGpu(0);
    let artifact_dir = "artifacts/en-pt_BR";
    let model = NMTConfig::new(N_EMBEDDINGS, D_MODEL);
    println!("{model}");
    println!("Training on device: {:?}", device);
    println!("Artifact directory: {}", artifact_dir);
    println!("Starting train...");
    let config = TrainingConfig {
        model: model,
        optimizer: AdamConfig::new().with_beta_2(0.98),
        num_epochs: 10,
        batch_size: 16,
        num_workers: 4,
        seed: 42,
        learning_rate: 5e-4,
    };
    train::<MyAutodiffBackend>(artifact_dir, config, device);

    Ok(())
}
