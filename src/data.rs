use burn::{data::dataloader::batcher::Batcher, prelude::*};

#[derive(Clone, Default)]
pub struct SpectraBatcher {}

#[derive(Clone, Debug)]
pub struct SpectraBatch<B: Backend> {
    pub spectra: Tensor<B, 2>,
    pub targets: Tensor<B, 2, Bool>,
}

pub const BIN_SIZE: usize = 4096;
pub const NUMBER_OF_ATOMS: usize = 90;

#[derive(Clone)]
struct ProcessedSpectrum {
    spectrum: [f64; BIN_SIZE],
    atom_present: [bool; NUMBER_OF_ATOMS],
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
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 2, Bool>::from_data(item.atom_present, device))
            .collect();

        let spectra = Tensor::cat(spectra, 0);
        let targets = Tensor::cat(targets, 0);

        SpectraBatch { spectra, targets }
    }
}
