use burn::{
    nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu, Sigmoid},
    prelude::*,
};

use crate::data::BIN_SIZE;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    batch_norm1: BatchNorm<B>,
    linear2: Linear<B>,
    batch_norm2: BatchNorm<B>,
    linear3: Linear<B>,
    dropout: Dropout,
    inner_activation: Relu,
    activation: Sigmoid,
    class_weights: Option<Vec<f32>>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.1")]
    dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        class_weights: Option<Vec<f32>>,
    ) -> Model<B> {
        Model {
            linear1: LinearConfig::new(BIN_SIZE, self.hidden_size).init(device),
            batch_norm1: BatchNormConfig::new(self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size / 2).init(device),
            batch_norm2: BatchNormConfig::new(self.hidden_size / 2).init(device),
            linear3: LinearConfig::new(self.hidden_size / 2, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Sigmoid::new(),
            inner_activation: Relu::new(),
            class_weights: class_weights,
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Spectra [batch_size, binned_spectrum_size]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, spectra: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, binned_spectrum_size] = spectra.dims();

        let x = spectra.reshape([batch_size, binned_spectrum_size]);
        let x = self.linear1.forward(x);
        let x = self.inner_activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.batch_norm1.forward(x);
        let x = self.linear2.forward(x);
        let x = self.inner_activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.batch_norm2.forward(x);
        let x = self.linear3.forward(x);
        self.activation.forward(x)
    }

    pub(crate) fn class_weights(&self) -> Option<Vec<f32>> {
        self.class_weights.clone()
    }
}
