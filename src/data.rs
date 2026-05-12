use burn::{data::dataloader::batcher::Batcher, prelude::*};
use mascot_rs::{mascot_generic_format::MGFVec, prelude::Spectrum};
use mass_spectrometry::traits::SpectrumAlloc;
use molecular_formulas::prelude::*;

use crate::error::TrainingError;

#[derive(Clone, Default)]
pub struct SpectraBatcher {}

#[derive(Clone, Debug)]
pub struct SpectraBatch<B: Backend> {
    pub spectra: Tensor<B, 2>,
    pub targets: Tensor<B, 2, Int>,
}

pub const TOP_K_PEAKS: usize = 64;
pub const BIN_SIZE: usize = 1000;
pub const NUMBER_OF_ATOMS: usize = ELEMENTS.len();
pub const ELEMENTS: &[Element; 18] = &[
    Element::H,
    Element::B,
    Element::C,
    Element::N,
    Element::O,
    Element::F,
    Element::Na,
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
    pub(crate) spectrum: [f32; TOP_K_PEAKS * 2],
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
            .map(|data| Tensor::<B, 1>::from_floats(data.spectrum, device))
            .map(|tensor| tensor.reshape([1, TOP_K_PEAKS * 2]))
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Bool>::from_bool(item.atom_present.into(), device))
            .map(|tensor| tensor.reshape([1, NUMBER_OF_ATOMS]).int())
            .collect();

        let spectra = Tensor::cat(spectra, 0);
        let targets = Tensor::cat(targets, 0);

        SpectraBatch { spectra, targets }
    }
}

// pub fn load_processed_spectra() -> Result<Vec<ProcessedSpectrum>, TrainingError> {
// let load = pollster::block_on(
// MGFVec::<f32>::annotated_ms2()
// .target_directory("data")
// .load(),
// )?;
// let mut output: Vec<ProcessedSpectrum> = Vec::with_capacity(load.spectra().len());
// for s in load.spectra() {
// let Some(formula) = s.metadata().formula() else {
// continue;
// };
//
// output.push(ProcessedSpectrum {
// spectrum: *s
// .linear_binned_intensities(0.0, 1000.0, BIN_SIZE)?
// .as_array::<BIN_SIZE>()
// .unwrap(),
// atom_present: *to_binary_vec(formula)?
// .as_array::<NUMBER_OF_ATOMS>()
// .unwrap(),
// });
// }
// Ok(output)
// }

pub fn load_processed_spectra() -> Result<Vec<ProcessedSpectrum>, TrainingError> {
    let load = pollster::block_on(
        MGFVec::<f32>::annotated_ms2()
            .target_directory("data")
            .load(),
    )?;
    let mut output: Vec<ProcessedSpectrum> = Vec::with_capacity(load.spectra().len());
    for s in load.spectra() {
        let Some(formula) = s.metadata().formula() else {
            continue;
        };

        let mut peaks = s
            .top_k_peaks(TOP_K_PEAKS)
            .unwrap()
            .peaks()
            .map(|(mz, int)| [mz, int])
            .collect::<Vec<[f32; 2]>>();

        // normalize the intensities
        let max_intensity = peaks.iter().map(|&[_mz, int]| int).fold(0.0, f32::max);
        for peak in peaks.iter_mut() {
            peak[1] /= max_intensity;
        }

        if peaks.len() < TOP_K_PEAKS {
            peaks.resize(TOP_K_PEAKS, [0.0, 0.0]);
        }

        let peaks = *peaks.as_array::<TOP_K_PEAKS>().unwrap();

        // we flatten the peaks into a single array of size 2*top peaks
        let mut final_peaks = [0.0; 2 * TOP_K_PEAKS];
        for (i, [mz, int]) in peaks.iter().enumerate() {
            final_peaks[2 * i] = *mz;
            final_peaks[2 * i + 1] = *int;
        }

        output.push(ProcessedSpectrum {
            spectrum: final_peaks,
            atom_present: *to_binary_vec(formula)?
                .as_array::<NUMBER_OF_ATOMS>()
                .unwrap(),
        });
    }
    Ok(output)
}

pub fn get_class_weights(data: &[ProcessedSpectrum]) -> Vec<f32> {
    let mut output: Vec<f32> = vec![0.0; NUMBER_OF_ATOMS];
    let n_samples = data.len() as f32;
    let n_classes = NUMBER_OF_ATOMS as f32;

    for d in data {
        for (i, &element_is_present) in d.atom_present.iter().enumerate() {
            if element_is_present {
                output[i] += 1.0;
            }
        }
    }

    for weight in output.iter_mut() {
        *weight = n_samples / (*weight * n_classes);
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

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use burn::{Tensor, backend::Metal};
    #[test]
    fn example() {
        type MyBackend = Metal<f32, i32>;
        let device = Default::default();
        let bool_tensor =
            Tensor::<MyBackend, 1, Bool>::from_bool([true, false, true].into(), &device);
        let int_tensor = bool_tensor.int();
        println!("{int_tensor}"); // [1, 0, 1]
    }

    #[test]
    fn example_2() {
        type MyBackend = Metal<f32, i32>;
        let device = Default::default();
        let tensor = Tensor::<MyBackend, 2>::from_data([[3.0]], &device);
        // Convert the tensor with a single element into a scalar.
        let scalar = tensor.into_scalar();
        println!("{scalar}");
    }
}
