use crate::model::Model;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::prelude::*;
use burn::train::MultiLabelClassificationOutput;

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        spectra: Tensor<B, 2>,
        targets: Tensor<B, 2, Int>,
    ) -> MultiLabelClassificationOutput<B> {
        let output = self.forward(spectra);
        let loss = BinaryCrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        MultiLabelClassificationOutput::new(loss, output, targets)
    }
}
