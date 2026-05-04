#![recursion_limit = "256"]
mod data;
mod model;
mod training;
use crate::{data::NUMBER_OF_ATOMS, model::ModelConfig};
use burn::backend::Metal;

fn main() {
    type MyBackend = Metal;

    let device = Default::default();
    let model = ModelConfig::new(NUMBER_OF_ATOMS, 512).init::<MyBackend>(&device, None);

    println!("{model}");
}
