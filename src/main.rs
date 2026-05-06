#![recursion_limit = "256"]
mod data;
mod mcc;
// mod metric;
mod model;
mod training;
use crate::{data::NUMBER_OF_ATOMS, model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Metal},
    optim::AdamConfig,
};
mod dataset;
mod error;

fn main() {
    type MyBackend = Metal<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./first_attempt";

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(NUMBER_OF_ATOMS, 256), AdamConfig::new()),
        device.clone(),
    );
}
