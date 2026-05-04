use crate::data::{SpectraBatch, SpectraBatcher};
use crate::model::{Model, ModelConfig};
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{InferenceStep, MultiLabelClassificationOutput, TrainOutput, TrainStep};

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        spectra: Tensor<B, 2>,
        targets: Tensor<B, 2, Int>,
    ) -> MultiLabelClassificationOutput<B> {
        let output = self.forward(spectra);
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_weights(self.class_weights())
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        MultiLabelClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = SpectraBatch<B>;
    type Output = MultiLabelClassificationOutput<B>;
    fn step(&self, batch: Self::Input) -> burn::train::TrainOutput<Self::Output> {
        let item = self.forward_classification(batch.spectra, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = SpectraBatch<B>;
    type Output = MultiLabelClassificationOutput<B>;
    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_classification(batch.spectra, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    B::seed(&device, config.seed);

    let batcher = SpectraBatcher::default();

    todo!()
}
