#![recursion_limit = "256"]

mod data;
mod inference;
mod mcc;
mod model;
mod training;
use crate::{
    data::NUMBER_OF_ATOMS, dataset::SpectraDataset, error::TrainingError, model::ModelConfig,
    training::TrainingConfig,
};
use burn::{
    backend::{Autodiff, Metal},
    optim::AdamConfig,
};
mod dataset;
mod error;

fn main() -> Result<(), TrainingError> {
    type MyBackend = Metal<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./first_attempt";

    println!("Loading spectra.");
    let dataset = SpectraDataset::new()?;
    println!("Finished loading spectra");
    let model_config = ModelConfig::new(NUMBER_OF_ATOMS, 256)
        .with_class_weights(Some(dataset.class_weights.clone()));

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        &dataset,
        TrainingConfig::new(model_config, AdamConfig::new()),
        device.clone(),
    );

    crate::inference::infer::<MyBackend>(artifact_dir, device, dataset.dataset[0].clone());
    Ok(())
}
