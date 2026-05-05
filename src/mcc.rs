use burn::prelude::*;
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

        // Flatten tensors for easier processing
        let preds = preds.reshape([-1]);
        let targets = targets.reshape([-1]);

        // Move to CPU data
        let preds_data = preds.to_data();
        let targets_data = targets.to_data();

        let preds_slice = preds_data.as_slice().unwrap();
        let targets_slice = targets_data.as_slice().unwrap();

        let mut tp: u64 = 0;
        let mut tn: u64 = 0;
        let mut fp: u64 = 0;
        let mut fn_: u64 = 0;

        for (p, t) in preds_slice.iter().zip(targets_slice.iter()) {
            match (p, t) {
                (1i64, 1i64) => tp += 1,
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
