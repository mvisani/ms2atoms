#![recursion_limit = "256"]

mod data;
mod inference;
mod mcc;
mod model;
mod training;
use std::fs::File;

use crate::{
    data::NUMBER_OF_ATOMS, dataset::SpectraDataset, error::TrainingError,
    inference::create_confusion_matrices, model::ModelConfig, training::TrainingConfig,
};
use burn::{
    backend::{Autodiff, Metal},
    optim::AdamConfig,
};
use csv::Writer;
mod dataset;
mod error;
mod output;

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

    let results =
        crate::inference::infer::<MyBackend>(artifact_dir, device, dataset.test(42).dataset);

    let confusion_matrices = create_confusion_matrices(results, dataset.test(42).dataset, 0.5);

    let file = File::create("results.csv")?;
    let mut wtr = Writer::from_writer(file);
    for matrix in confusion_matrices {
        wtr.serialize(matrix).unwrap();
    }

    wtr.flush()?;
    Ok(())
}
