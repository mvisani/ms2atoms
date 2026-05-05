//! Error handling for the Burn training CLI.

use std::path::PathBuf;

use thiserror::Error;

/// Top-level error for training data loading, model training, and evaluation.
#[derive(Debug, Error)]
pub enum TrainingError {
    /// Generic file I/O failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Required dataset split files were missing.
    #[error("missing dataset split file: {0}")]
    MissingFile(PathBuf),
    /// The dataset contents did not match the expected schema.
    #[error("invalid dataset: {0}")]
    Dataset(String),
    /// Burn training or recording failed.
    #[error("burn training failed: {0}")]
    Burn(String),
}
