use crate::{
    data::{ELEMENTS, NUMBER_OF_ATOMS, ProcessedSpectrum, SpectraBatcher},
    output::ConfusionMatrix,
    training::TrainingConfig,
};
use burn::{data::dataloader::batcher::Batcher, record::Recorder, tensor::Transaction};
use burn::{prelude::*, record::CompactRecorder};

pub fn infer<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
    items: Vec<ProcessedSpectrum>,
) -> Tensor<B, 2> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config
        .model
        .init::<B>(&device, config.model.class_weights())
        .load_record(record);

    let batcher = SpectraBatcher::default();
    let batch = batcher.batch(items, &device);
    model.forward(batch.spectra)
}

pub fn create_confusion_matrices<B: Backend>(
    predictions: Tensor<B, 2>,
    items: Vec<ProcessedSpectrum>,
    threshold: f64,
) -> Vec<ConfusionMatrix> {
    let mut confusion_matrices: Vec<ConfusionMatrix> = Vec::with_capacity(NUMBER_OF_ATOMS);

    for element in ELEMENTS {
        confusion_matrices.push(ConfusionMatrix::new(element));
    }
    let predicted_tensor = predictions.greater_elem(threshold);
    let pred_iter = predicted_tensor.iter_dim(0);

    for (p, t) in pred_iter.into_iter().zip(items.iter()) {
        let [output_data] = Transaction::default()
            .register(p)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        for (i, (predicted_atom, true_atom)) in (output_data
            .as_slice::<u8>()
            .unwrap()
            .iter()
            .zip(t.atom_present))
        .enumerate()
        {
            match (*predicted_atom, true_atom) {
                (1, true) => confusion_matrices[i].true_positive += 1,
                (0, false) => confusion_matrices[i].true_negative += 1,
                (1, false) => confusion_matrices[i].false_positive += 1,
                (0, true) => confusion_matrices[i].false_negative += 1,
                _ => unreachable!("All possibilites should have matched !"),
            }
        }
    }

    confusion_matrices
}
