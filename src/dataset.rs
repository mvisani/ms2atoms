use burn::data::dataset::{
    Dataset, InMemDataset,
    transform::{PartialDataset, ShuffledDataset},
};

use crate::data::{ProcessedSpectrum, load_processed_spectra};

type ShuffledSpectraDataset = ShuffledDataset<InMemDataset<ProcessedSpectrum>, ProcessedSpectrum>;

pub struct SpectraDataset {
    dataset: PartialDataset<ShuffledSpectraDataset, ProcessedSpectrum>,
}

impl SpectraDataset {
    /// Creates a new train dataset.
    pub fn train(seed: u64) -> Self {
        Self::new("train", seed)
    }

    /// Creates a new test dataset.
    pub fn test(seed: u64) -> Self {
        Self::new("test", seed)
    }

    fn new(split: &str, seed: u64) -> Self {
        let vec_of_data = load_processed_spectra().unwrap();
        let vec_of_data = InMemDataset::new(vec_of_data);
        let dataset: ShuffledSpectraDataset = ShuffledDataset::new(vec_of_data, seed);
        let len = dataset.len();
        let data_split: PartialDataset<
            ShuffledDataset<InMemDataset<ProcessedSpectrum>, ProcessedSpectrum>,
            ProcessedSpectrum,
        > = match split {
            "train" => PartialDataset::new(dataset, 0, len * 8 / 10),
            "test" => PartialDataset::new(dataset, len * 8 / 10, len),
            _ => panic!("Invalid split type"),
        };

        Self {
            dataset: data_split,
        }
    }
}

impl Dataset<ProcessedSpectrum> for SpectraDataset {
    fn get(&self, index: usize) -> Option<ProcessedSpectrum> {
        self.dataset.get(index)
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}
