use crate::data::{SpectraBatch, SpectraBatcher};
use crate::dataset::SpectraDataset;
use crate::metrics::mcc::MatthewsCorrelationMetric;
use crate::model::{Model, ModelConfig};
use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{HammingScore, LossMetric};
use burn::train::{
    InferenceStep, Learner, MetricEarlyStoppingStrategy, MultiLabelClassificationOutput,
    StoppingCondition, SupervisedTraining, TrainOutput, TrainStep,
};

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        spectra: Tensor<B, 2>,
        targets: Tensor<B, 2, Int>,
    ) -> MultiLabelClassificationOutput<B> {
        let logits = self.forward_logit(spectra);
        let outputs = self.activation.forward(logits.clone());
        let loss_bce = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .with_weights(self.class_weights())
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());

        let lambda = 1e-3;
        let logit_reg = logits.clone().powf_scalar(2.0).mean();
        let loss = loss_bce + logit_reg * lambda;
        MultiLabelClassificationOutput::new(loss, outputs, targets)
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
    #[config(default = 5)]
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

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    dataset: &SpectraDataset,
    config: TrainingConfig,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    B::seed(&device, config.seed);

    let batcher = SpectraBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset.train(config.seed));

    let dataloader_test = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset.test(config.seed));

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((
            MatthewsCorrelationMetric::new(),
            // MaxLogit::new(),
            LossMetric::new(),
            HammingScore::new(),
        ))
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &LossMetric::<B>::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 2 },
        ))
        .num_epochs(config.num_epochs)
        .summary();

    let model = config
        .model
        .init::<B>(&device, Some(dataset.class_weights.clone()));
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
