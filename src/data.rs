use burn::{data::dataloader::batcher::Batcher, prelude::*};

#[derive(Clone, Default)]
pub struct Batcher {}

#[derive(Clone, Debug)]
pub struct Batch<B: Backend> {
    pub spectra: Tensor<B, 2>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, I, Batch<B>> for Batcher {
    fn batch(&self, items: Vec<I>, device: &<B as Backend>::Device) -> O {}
}
