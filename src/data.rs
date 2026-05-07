use burn::{data::dataloader::batcher::Batcher, prelude::*};
use mascot_rs::{mascot_generic_format::MGFVec, prelude::Spectrum};
use molecular_formulas::prelude::*;

use crate::error::TrainingError;

#[derive(Clone, Default)]
pub struct SpectraBatcher {}

#[derive(Clone, Debug)]
pub struct SpectraBatch<B: Backend> {
    pub spectra: Tensor<B, 2>,
    pub targets: Tensor<B, 2, Int>,
}

pub const BIN_SIZE: usize = 8192;
pub const NUMBER_OF_ATOMS: usize = ELEMENTS.len();
pub const ELEMENTS: &[Element; 19] = &[
    Element::H,
    Element::B,
    Element::C,
    Element::N,
    Element::O,
    Element::F,
    Element::Na,
    Element::Mg,
    Element::Si,
    Element::P,
    Element::S,
    Element::Cl,
    Element::K,
    Element::Fe,
    Element::Co,
    Element::As,
    Element::Se,
    Element::Br,
    Element::I,
];

#[derive(Clone, Debug)]
pub struct ProcessedSpectrum {
    pub(crate) spectrum: [f64; BIN_SIZE],
    pub(crate) atom_present: [bool; NUMBER_OF_ATOMS],
}

impl<B: Backend> Batcher<B, ProcessedSpectrum, SpectraBatch<B>> for SpectraBatcher {
    fn batch(
        &self,
        items: Vec<ProcessedSpectrum>,
        device: &<B as Backend>::Device,
    ) -> SpectraBatch<B> {
        let spectra = items
            .iter()
            .map(|item| TensorData::from(item.spectrum).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, BIN_SIZE]))
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Bool>::from_data(item.atom_present, device))
            .map(|tensor| tensor.reshape([1, NUMBER_OF_ATOMS]).int())
            .collect();

        let spectra = Tensor::cat(spectra, 0);
        let targets = Tensor::cat(targets, 0);

        SpectraBatch { spectra, targets }
    }
}

pub fn load_processed_spectra() -> Result<Vec<ProcessedSpectrum>, TrainingError> {
    let load = pollster::block_on(
        MGFVec::<f64>::annotated_ms2()
            .target_directory("data")
            .load(),
    )?;
    let mut output: Vec<ProcessedSpectrum> = Vec::with_capacity(load.spectra().len());
    for s in load.spectra() {
        let Some(formula) = s.metadata().formula() else {
            continue;
        };

        output.push(ProcessedSpectrum {
            spectrum: *s
                .linear_binned_intensities(0.0, 1000.0, BIN_SIZE)?
                .as_array::<BIN_SIZE>()
                .unwrap(),
            atom_present: *to_binary_vec(formula)?
                .as_array::<NUMBER_OF_ATOMS>()
                .unwrap(),
        });
    }
    Ok(output)
}

pub fn get_class_weights(data: &[ProcessedSpectrum]) -> Vec<f32> {
    let mut output: Vec<f32> = vec![0.0; NUMBER_OF_ATOMS];
    let n = data.len() as f32;

    for d in data {
        for (i, &element_is_present) in d.atom_present.iter().enumerate() {
            if element_is_present {
                output[i] += 1.0;
            }
        }
    }

    for weight in output.iter_mut() {
        let freq = *weight / n; // frequency in [0, 1]
        *weight = (1.0 - freq).max(1e-6); // stays in [0, 1]
    }

    output
}

fn to_binary_vec(
    formula: &ChemicalFormula<u32, i32>,
) -> Result<[bool; NUMBER_OF_ATOMS], TrainingError> {
    let mut binary_count = [false; NUMBER_OF_ATOMS];
    for (i, &e) in ELEMENTS.iter().enumerate() {
        if formula.contains_element(e) {
            binary_count[i] = true;
        }
    }
    Ok(binary_count)
}

fn to_count_vec(
    formula: &ChemicalFormula<u32, i32>,
) -> Result<[i32; NUMBER_OF_ATOMS], TrainingError> {
    let mut binary_count = [0; NUMBER_OF_ATOMS];
    for (i, &e) in ELEMENTS.iter().enumerate() {
        if formula.contains_element(e) {
            binary_count[i] += formula.count_of_element::<u32>(e)? as i32
        }
    }
    Ok(binary_count)
}
