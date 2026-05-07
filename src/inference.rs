use crate::{
    data::{ProcessedSpectrum, SpectraBatcher},
    training::TrainingConfig,
};
use burn::{data::dataloader::batcher::Batcher, record::Recorder};
use burn::{prelude::*, record::CompactRecorder};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: ProcessedSpectrum) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config
        .model
        .init::<B>(&device, config.model.class_weights())
        .load_record(record);

    let label = item.atom_present;
    let batcher = SpectraBatcher::default();
    let batch = batcher.batch(vec![item], &device);
    let output = model.forward(batch.spectra);
    let predicted = output.flatten::<1>(0, 1);

    println!("Predicted {predicted} Expected {label:#?}");
}
