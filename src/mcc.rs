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

impl<B: Backend> Metric for MatthewsCorrelationMetric<B>
where
    B::FloatElem: Into<f64>,
{
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
        let preds_f = outputs.greater_elem(self.threshold).float();

        let targets_f = targets.float();

        let ones = Tensor::<B, 2>::ones_like(&preds_f);

        let tp = (preds_f.clone() * targets_f.clone()).sum_dim(1);

        let tn = ((ones.clone() - preds_f.clone()) * (ones.clone() - targets_f.clone())).sum_dim(1);

        let fp = (preds_f.clone() * (ones.clone() - targets_f.clone())).sum_dim(1);

        let fn_ = ((ones - preds_f) * targets_f).sum_dim(1);

        let numerator = tp.clone() * tn.clone() - fp.clone() * fn_.clone();

        let denominator = ((tp.clone() + fp.clone())
            * (tp.clone() + fn_.clone())
            * (tn.clone() + fp)
            * (tn + fn_))
            .sqrt();

        let mcc = numerator / denominator.clamp_min(1e-12);

        // Average batch MCC
        let mcc_value = mcc.mean().into_scalar();

        // Update state
        self.state.update(
            mcc_value.into(),
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
            unit: None,
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
