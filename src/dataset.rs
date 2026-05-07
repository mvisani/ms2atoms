use rand::prelude::*;
use rand::seq::SliceRandom;

use burn::data::dataset::Dataset;
use rand::rngs::ChaCha8Rng;

use crate::{
    data::{ProcessedSpectrum, get_class_weights, load_processed_spectra},
    error::TrainingError,
};

pub struct SpectraDataset {
    pub(crate) dataset: Vec<ProcessedSpectrum>,
    pub(crate) class_weights: Vec<f32>,
}

impl SpectraDataset {
    /// Creates a new train dataset.
    pub fn train(&self, seed: u64) -> Self {
        let mut vec_of_data = self.dataset.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        vec_of_data.shuffle(&mut rng);

        let len = vec_of_data.len();
        let Some(subset) = vec_of_data.get(0..(len * 8 / 10)) else {
            unreachable!("There was a problem subsetting the vector")
        };

        Self {
            dataset: subset.to_vec(),
            class_weights: self.class_weights.clone(),
        }
    }

    /// Creates a new test dataset.
    pub fn test(&self, seed: u64) -> Self {
        let mut vec_of_data = self.dataset.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        vec_of_data.shuffle(&mut rng);

        let len = vec_of_data.len();
        let Some(subset) = self.dataset.get((len * 8 / 10)..len) else {
            unreachable!("There was a problem subsetting the vector")
        };

        Self {
            dataset: subset.to_vec(),
            class_weights: self.class_weights.clone(),
        }
    }

    pub fn new() -> Result<Self, TrainingError> {
        let vec_of_data = load_processed_spectra()?;
        let weights = get_class_weights(&vec_of_data);

        Ok(Self {
            dataset: vec_of_data,
            class_weights: weights,
        })
    }
}

impl Dataset<ProcessedSpectrum> for SpectraDataset {
    fn get(&self, index: usize) -> Option<ProcessedSpectrum> {
        self.dataset.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}
