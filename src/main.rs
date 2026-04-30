#![recursion_limit = "256"]
mod data;
mod model;
mod training;
use crate::{data::BIN_SIZE, model::ModelConfig};
use burn::backend::Metal;

fn main() {
    type MyBackend = Metal;

    let device = Default::default();
    let model = ModelConfig::new(10, BIN_SIZE).init::<MyBackend>(&device);

    println!("{model}");
}
