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
pub struct MaxLogit<B: Backend> {
    name: Arc<String>,
    state: NumericMetricState,
    _b: PhantomData<B>,
}

/// The [Max Logit](MaxLogit) input type.
pub struct MaxLogitInput<B: Backend> {
    outputs: Tensor<B, 2>,
}

impl<B: Backend> MaxLogit<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Default for MaxLogit<B> {
    /// Creates a new metric instance with default values.
    fn default() -> Self {
        let name = Arc::new("Max Logit".into());

        Self {
            name,
            state: NumericMetricState::default(),
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for MaxLogit<B>
where
    B::FloatElem: Into<f64>,
{
    type Input = MaxLogitInput<B>;
    fn update(
        &mut self,
        input: &Self::Input,
        _metadata: &burn::train::metric::MetricMetadata,
    ) -> burn::train::metric::SerializedEntry {
        let [batch_size, _n_classes] = input.outputs.dims();
        let outputs = input.outputs.clone();

        let max_logits = outputs.max().into_scalar();
        // Update state
        self.state.update(
            max_logits.into(),
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

impl<B: Backend> Numeric for MaxLogit<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

impl<B: Backend> Adaptor<MaxLogitInput<B>> for MultiLabelClassificationOutput<B> {
    fn adapt(&self) -> MaxLogitInput<B> {
        MaxLogitInput {
            outputs: self.output.clone(),
        }
    }
}
