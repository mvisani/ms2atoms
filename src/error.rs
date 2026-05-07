//! Error handling for the Burn training CLI.

use mascot_rs::error::MascotError;

use mass_spectrometry::structs::SimilarityComputationError;
use molecular_formulas::errors::{CountError, NumericError, ParserError};
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
    /// MolecularFormula Count error
    #[error(transparent)]
    MolecularFormulaCount(#[from] CountError),
    /// MolecularFormula Numeric error
    #[error(transparent)]
    MolecularFormulaNumeric(#[from] NumericError),
    /// MolecularFormula parse error
    #[error(transparent)]
    MolecularFormulaParse(#[from] ParserError),
    /// Mascot-rs error
    #[error(transparent)]
    Mascot(#[from] MascotError),
    /// Mass-spec traits similarity computation errors
    #[error(transparent)]
    MassSpecTraitSimilarityComputation(#[from] SimilarityComputationError),
}
