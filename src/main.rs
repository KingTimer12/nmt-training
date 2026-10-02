use std::any::Any;
use std::panic::{self, AssertUnwindSafe};

use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::optim::AdamConfig;
use burn::tensor::backend::AutodiffBackend;

use crate::backend::Selected;
use crate::model::config::NMTConfig;
use crate::data::dataset::Corpus;
use crate::training::{TrainingConfig, train};

mod backend;
#[cfg(feature = "bundle")]
mod bundle;
mod data;
pub mod model;
mod tokenizer;
mod training;
pub mod utils;

pub const CORPUS: &str = "data/canonical/en-pt_BR/tatoeba.jsonl";

const D_MODEL: usize = 256; // quanto maior, mais pesado e lento será o treinamento, mas melhor será a qualidade da tradução

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "bundle")]
    bundle::prepare()?;

    let selected = backend::select();
    println!("Selected backend: {selected}");

    match selected {
        #[cfg(target_os = "linux")]
        Selected::Cuda(device) => run::<Autodiff<burn::backend::Cuda<f32, i32>>>(device),
        #[cfg(target_os = "linux")]
        Selected::Rocm(device) => run::<Autodiff<burn::backend::Rocm<f32, i32>>>(device),
        Selected::Wgpu(device) => run::<Autodiff<Wgpu<f32, i32>>>(device),
        Selected::Cpu => run::<Autodiff<NdArray<f32, i32>>>(Default::default()),
    }

    Ok(())
}

/// Batch sizes tried in order: on out-of-memory the run restarts with the next one.
const BATCH_SIZES: [usize; 8] = [128, 64, 32, 16, 8, 4, 2, 1];

fn run<B: AutodiffBackend>(device: B::Device) {
    let artifact_dir = "artifacts/en-pt_BR";
    // Loaded once and reused by every out-of-memory retry.
    let corpus = Corpus::load();
    let model = NMTConfig::new(corpus.vocab.len(), D_MODEL);
    println!("{model}");
    println!("Training on device: {:?}", device);
    println!("Artifact directory: {}", artifact_dir);

    for batch_size in BATCH_SIZES {
        println!("Starting train with batch_size={batch_size}...");
        let config = TrainingConfig {
            model: model.clone(),
            optimizer: AdamConfig::new().with_beta_2(0.98),
            num_epochs: 50,
            batch_size,
            num_workers: 4,
            seed: 42,
            learning_rate: 5e-4,
            tokens_per_item: 24,
            early_stopping_patience: 5,
        };
        let device = device.clone();
        match panic::catch_unwind(AssertUnwindSafe(|| {
            train::<B>(artifact_dir, config, &corpus, device)
        })) {
            Ok(()) => return,
            Err(payload) if is_out_of_memory(payload.as_ref()) => {
                eprintln!("Out of memory with batch_size={batch_size}, retrying smaller...");
            }
            Err(payload) => panic::resume_unwind(payload),
        }
    }
    panic!("Out of memory even with batch_size=1");
}

fn is_out_of_memory(payload: &(dyn Any + Send)) -> bool {
    let msg = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or_default()
        .to_lowercase();
    ["out of memory", "out-of-memory", "outofmemory", "out_of_memory", "alloc"]
        .iter()
        .any(|k| msg.contains(k))
}
