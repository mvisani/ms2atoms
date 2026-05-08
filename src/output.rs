use elements_rs::Element;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct ConfusionMatrix {
    pub(crate) element: String,
    pub(crate) true_positive: u32,
    pub(crate) true_negative: u32,
    pub(crate) false_positive: u32,
    pub(crate) false_negative: u32,
}

impl ConfusionMatrix {
    pub fn new(element: &Element) -> Self {
        Self {
            element: element.symbol().to_string(),
            true_positive: 0,
            true_negative: 0,
            false_positive: 0,
            false_negative: 0,
        }
    }
}
