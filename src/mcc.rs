use burn::prelude::*;
use burn::tensor::Transaction;
use burn::tensor::activation::sigmoid;
use burn::train::MultiLabelClassificationOutput;
use burn::train::metric::Adaptor;
use burn::train::metric::Numeric;
use burn::train::metric::NumericAttributes;
use burn::train::metric::NumericEntry;
use burn::train::metric::state::FormatOptions;
use burn::{
    Tensor,
    tensor::backend::Backend,
    train::metric::{Metric, state::NumericMetricState},
};
use core::marker::PhantomData;
use std::sync::Arc;

/// The mcc metric.
#[derive(Clone)]
pub struct MatthewsCorrelationMetric<B: Backend> {
    name: Arc<String>,
    state: NumericMetricState,
    threshold: f32,
    sigmoid: bool,
    _b: PhantomData<B>,
}

/// The [MCC metric](MatthewsCorrelationMetric) input type.
pub struct MCCInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B, 2, Int>,
}

impl<B: Backend> MatthewsCorrelationMetric<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self::default()
    }

    fn update_name(&mut self) {
        self.name = Arc::new(format!("MCC @ Threshold({})", self.threshold));
    }

    /// Sets the threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self.update_name();
        self
    }

    /// Sets the sigmoid activation function usage.
    pub fn with_sigmoid(mut self, sigmoid: bool) -> Self {
        self.sigmoid = sigmoid;
        self.update_name();
        self
    }
}

impl<B: Backend> Default for MatthewsCorrelationMetric<B> {
    /// Creates a new metric instance with default values.
    fn default() -> Self {
        let threshold = 0.5;
        let name = Arc::new(format!("MCC Score @ Threshold({})", threshold));

        Self {
            name,
            state: NumericMetricState::default(),
            threshold,
            sigmoid: false,
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for MatthewsCorrelationMetric<B> {
    type Input = MCCInput<B>;
    fn update(
        &mut self,
        input: &Self::Input,
        _metadata: &burn::train::metric::MetricMetadata,
    ) -> burn::train::metric::SerializedEntry {
        let [batch_size, _n_classes] = input.outputs.dims();
        let targets = input.targets.clone();
        let mut outputs = input.outputs.clone();

        if self.sigmoid {
            outputs = sigmoid(outputs);
        }

        // Apply threshold -> predictions {0,1}
        let preds = outputs.greater_elem(self.threshold).int();

        let preds_iter = preds.iter_dim(0);
        let targets_iter = targets.iter_dim(0);

        let mut mcc_sum = 0.0;
        let mut size = 0;

        for (p, t) in preds_iter.into_iter().zip(targets_iter.into_iter()) {
            let [output_data, targets_data] = Transaction::default()
                .register(p)
                .register(t)
                .execute()
                .try_into()
                .expect("Correct amount of tensor data");
            mcc_sum += calculate_mcc(
                output_data.as_slice().unwrap(),
                targets_data.as_slice().unwrap(),
            );
            size += 1;
        }

        let mcc: f64 = mcc_sum / (size as f64);

        // Update state
        self.state.update(
            mcc,
            batch_size,
            FormatOptions::new(self.name()).precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> burn::train::metric::MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> burn::train::metric::MetricAttributes {
        NumericAttributes {
            unit: Some("".to_string()),
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend> Numeric for MatthewsCorrelationMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

impl<B: Backend> Adaptor<MCCInput<B>> for MultiLabelClassificationOutput<B> {
    fn adapt(&self) -> MCCInput<B> {
        MCCInput {
            outputs: self.output.clone(),
            targets: self.targets.clone(),
        }
    }
}

fn calculate_mcc(predictions: &[i64], targets: &[i64]) -> f64 {
    let mut tp: u64 = 0;
    let mut tn: u64 = 0;
    let mut fp: u64 = 0;
    let mut fn_: u64 = 0;

    for (p, t) in predictions.iter().zip(targets.iter()) {
        match (p, t) {
            (1, 1) => tp += 1,
            (0, 0) => tn += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_ += 1,
            _ => {}
        }
    }

    let numerator = (tp * tn) as f64 - (fp * fn_) as f64;
    let denominator =
        ((tp + fp) as f64 * (tp + fn_) as f64 * (tn + fp) as f64 * (tn + fn_) as f64).sqrt();

    let mcc = if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    };
    mcc
}
