use burn::{data::dataloader::batcher::Batcher, prelude::*};
use mascot_rs::{mascot_generic_format::MGFVec, prelude::Spectrum};
use molecular_formulas::prelude::*;
use std::{
    path::{Path, PathBuf},
    str::FromStr,
};
use zenodo_rs::{Auth, RecordId, ZenodoClient};

#[derive(Clone, Default)]
pub struct SpectraBatcher {}

#[derive(Clone, Debug)]
pub struct SpectraBatch<B: Backend> {
    pub spectra: Tensor<B, 2>,
    pub targets: Tensor<B, 2, Int>,
}

pub const BIN_SIZE: usize = 4096;
pub const NUMBER_OF_ATOMS: usize = ELEMENTS.len();
pub const DEFAULT_DATA_DIR: &str = "data";
pub const DATASET_RECORD_ID: u64 = 19217442;
pub const FILE_NAME: &str = "clean_spectra.mgf";
pub const MASSSPECGYM_SPECTRA: usize = 231104;
pub const ELEMENTS: &[Element; 92] = &[
    Element::H,
    Element::He,
    Element::Li,
    Element::Be,
    Element::B,
    Element::C,
    Element::N,
    Element::O,
    Element::F,
    Element::Ne,
    Element::Na,
    Element::Mg,
    Element::Al,
    Element::Si,
    Element::P,
    Element::S,
    Element::Cl,
    Element::Ar,
    Element::K,
    Element::Ca,
    Element::Sc,
    Element::Ti,
    Element::V,
    Element::Cr,
    Element::Mn,
    Element::Fe,
    Element::Co,
    Element::Ni,
    Element::Cu,
    Element::Zn,
    Element::Ga,
    Element::Ge,
    Element::As,
    Element::Se,
    Element::Br,
    Element::Kr,
    Element::Rb,
    Element::Sr,
    Element::Y,
    Element::Zr,
    Element::Nb,
    Element::Mo,
    Element::Tc,
    Element::Ru,
    Element::Rh,
    Element::Pd,
    Element::Ag,
    Element::Cd,
    Element::In,
    Element::Sn,
    Element::Sb,
    Element::Te,
    Element::I,
    Element::Xe,
    Element::Cs,
    Element::Ba,
    Element::La,
    Element::Ce,
    Element::Pr,
    Element::Nd,
    Element::Pm,
    Element::Sm,
    Element::Eu,
    Element::Gd,
    Element::Tb,
    Element::Dy,
    Element::Ho,
    Element::Er,
    Element::Tm,
    Element::Yb,
    Element::Lu,
    Element::Hf,
    Element::Ta,
    Element::W,
    Element::Re,
    Element::Os,
    Element::Ir,
    Element::Pt,
    Element::Au,
    Element::Hg,
    Element::Tl,
    Element::Pb,
    Element::Bi,
    Element::Po,
    Element::At,
    Element::Rn,
    Element::Fr,
    Element::Ra,
    Element::Ac,
    Element::Th,
    Element::Pa,
    Element::U,
];

#[derive(Clone, Debug)]
pub struct ProcessedSpectrum {
    spectrum: [f32; BIN_SIZE],
    atom_present: [bool; NUMBER_OF_ATOMS],
}

impl<B: Backend> Batcher<B, ProcessedSpectrum, SpectraBatch<B>> for SpectraBatcher {
    fn batch(
        &self,
        items: Vec<ProcessedSpectrum>,
        device: &<B as Backend>::Device,
    ) -> SpectraBatch<B> {
        let spectra = items
            .iter()
            .map(|item| TensorData::from(item.spectrum).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 2, Int>::from_data(item.atom_present, device))
            .collect();

        let spectra = Tensor::cat(spectra, 0);
        let targets = Tensor::cat(targets, 0);

        SpectraBatch { spectra, targets }
    }
}

pub fn load_processed_spectra() -> Result<Vec<ProcessedSpectrum>, Box<dyn std::error::Error>> {
    let load = pollster::block_on(MGFVec::<usize, f32>::mass_spec_gym().load())?;
    let mut output: Vec<ProcessedSpectrum> = Vec::with_capacity(load.spectra().len());
    for s in load.spectra() {
        let formula_str = s.metadata().arbitrary_metadata_value("FORMULA").unwrap();
        let formula: ChemicalFormula<u16, i16> = ChemicalFormula::from_str(formula_str).unwrap();
        output.push(ProcessedSpectrum {
            spectrum: *s
                .linear_binned_intensities(0.0, 1000.0, BIN_SIZE)
                .unwrap()
                .as_array::<BIN_SIZE>()
                .unwrap(),
            atom_present: *to_binary_vec(formula)
                .as_array::<NUMBER_OF_ATOMS>()
                .unwrap(),
        });
    }
    Ok(output)
}

pub fn get_class_weights(data: &[ProcessedSpectrum]) -> Vec<f32> {
    let mut output: Vec<f32> = vec![0.0; NUMBER_OF_ATOMS];
    for d in data {
        for (i, &element_is_present) in d.atom_present.iter().enumerate() {
            if element_is_present {
                output[i] += 1.0
            }
        }
    }

    for mean in output.iter_mut() {
        *mean = 1.0 - *mean;
    }
    output
}

fn to_binary_vec(formula: ChemicalFormula) -> [bool; NUMBER_OF_ATOMS] {
    let mut binary_count = [false; NUMBER_OF_ATOMS];
    for (i, &e) in ELEMENTS.iter().enumerate() {
        if formula.contains_element(e) {
            binary_count[i] = true;
        }
    }
    binary_count
}

fn temporary_download_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("download");
    path.with_file_name(format!("{file_name}.part"))
}

fn download_data() -> Result<(), Box<dyn std::error::Error>> {
    let client = ZenodoClient::builder(Auth::new(
        std::env::var(Auth::TOKEN_ENV_VAR).unwrap_or_default(),
    ))
    .build()?;
    let data_dir = PathBuf::from(DEFAULT_DATA_DIR);
    std::fs::create_dir_all(&data_dir)?;
    let path = data_dir.join(FILE_NAME);

    let temp_path = temporary_download_path(&path);
    if temp_path.exists() {
        std::fs::remove_file(&temp_path)?;
    }

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(client.download_record_file_by_key_to_path(
            RecordId(DATASET_RECORD_ID),
            FILE_NAME,
            &temp_path,
        ))?;

    std::fs::rename(&temp_path, &path)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use elements_rs::{
        AtomicNumber, Element,
        isotopes::{HydrogenIsotope, Isotope},
    };
    use mascot_rs::prelude::*;
    use molecular_formulas::ChemicalFormula;

    use crate::data::{ELEMENTS, to_binary_vec};

    #[test]
    fn test_read_mgf() -> Result<()> {
        let spectra: MGFVec<usize> = MGFVec::from_path("data/clean_spectra.mgf")?;
        assert_eq!(
            spectra[0].metadata().arbitrary_metadata_value("FORMULA"),
            Some("C15H10ClF3N2O6S")
        );

        Ok(())
    }

    #[test]
    fn load_massspecgym() -> Result<()> {
        let load = pollster::block_on(MGFVec::<usize, f32>::mass_spec_gym().load())?;

        assert_eq!(load.spectra().len(), super::MASSSPECGYM_SPECTRA);
        let formula = load.spectra()[0]
            .metadata()
            .arbitrary_metadata_value("FORMULA");
        assert_eq!(formula, Some("C16H17NO4"));

        let formula: ChemicalFormula<u16, i16> =
            ChemicalFormula::from_str(formula.unwrap()).unwrap();

        let bool_vec = to_binary_vec(formula);
        let mut expected_result = [false; ELEMENTS.len()];
        expected_result[(Element::C.atomic_number() - 1) as usize] = true;
        expected_result[(Element::H.atomic_number() - 1) as usize] = true;
        expected_result[(Element::N.atomic_number() - 1) as usize] = true;
        expected_result[(Element::O.atomic_number() - 1) as usize] = true;

        assert_eq!(bool_vec, expected_result);

        Ok(())
    }
}
