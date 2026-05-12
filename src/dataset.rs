use rand::prelude::*;
use rand::seq::SliceRandom;

use burn::data::dataset::Dataset;
use rand::rngs::ChaCha8Rng;

use crate::{
    data::{NUMBER_OF_ATOMS, ProcessedSpectrum, get_class_weights, load_processed_spectra},
    error::TrainingError,
    output,
};

pub struct SpectraDataset {
    pub(crate) dataset: Vec<ProcessedSpectrum>,
    pub(crate) class_weights: Vec<f32>,
    pub(crate) rare_element_indices: Vec<Vec<usize>>,
}

impl SpectraDataset {
    /// Creates a new train dataset.
    pub fn train(&self, seed: u64) -> Self {
        Self {
            dataset: self.shuffle_and_split(seed, "train"),
            class_weights: self.class_weights.clone(),
            rare_element_indices: self.rare_element_indices.clone(),
        }
    }

    /// Creates a new test dataset.
    pub fn test(&self, seed: u64) -> Self {
        Self {
            dataset: self.shuffle_and_split(seed, "test"),
            class_weights: self.class_weights.clone(),
            rare_element_indices: self.rare_element_indices.clone(),
        }
    }

    fn shuffle_and_split(&self, seed: u64, split: &str) -> Vec<ProcessedSpectrum> {
        let mut indices = self.rare_element_indices.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut indices_to_get = Vec::new();

        for index_vec in indices.iter_mut() {
            index_vec.shuffle(&mut rng);

            let len = index_vec.len();
            let split_index = len * 2 / 3;

            let Some(subset) = index_vec.get(match split {
                "train" => 0..split_index,
                "test" => split_index..len,
                _ => unreachable!("Split should be either train or test"),
            }) else {
                unreachable!("There was a problem subsetting the vector")
            };

            indices_to_get.extend_from_slice(subset);
        }

        let mut output = Vec::with_capacity(indices_to_get.len());
        for index in indices_to_get {
            output.push(self.dataset[index].clone());
        }
        output
    }

    pub fn new() -> Result<Self, TrainingError> {
        let vec_of_data = load_processed_spectra()?;
        let weights = get_class_weights(&vec_of_data);

        let mut indices_split: Vec<Vec<usize>> = vec![Vec::new(); NUMBER_OF_ATOMS];

        for (i, data) in vec_of_data.iter().enumerate() {
            let least_frequent_class = find_least_frequent_class(data, &weights);
            indices_split[least_frequent_class].push(i);
        }

        Ok(Self {
            dataset: vec_of_data,
            class_weights: weights,
            rare_element_indices: indices_split,
        })
    }
}

fn find_least_frequent_class(data: &ProcessedSpectrum, class_weights: &[f32]) -> usize {
    // we know that the highest class weight corresponds to the least frequent class

    let mut least_frequent_class = 0;
    let mut highest_weight = 0.0;
    for (i, &is_present) in data.atom_present.iter().enumerate() {
        if is_present && class_weights[i] > highest_weight {
            highest_weight = class_weights[i];
            least_frequent_class = i;
        }
    }
    least_frequent_class
}

impl Dataset<ProcessedSpectrum> for SpectraDataset {
    fn get(&self, index: usize) -> Option<ProcessedSpectrum> {
        self.dataset.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}
