#![recursion_limit = "256"]
mod data;
mod model;
mod training;
use crate::model::ModelConfig;
use burn::backend::Metal;

fn main() {
    type MyBackend = Metal;

    let device = Default::default();
    let model = ModelConfig::new(10, 2048).init::<MyBackend>(&device);

    println!("{model}");
}
