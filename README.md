# Therapeutics Data Commons (TDC) Dataset Manager

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python tool for exploring, downloading, and preparing datasets from the Therapeutics Data Commons (TDC) for machine learning applications.

## About Therapeutics Data Commons

The Therapeutics Data Commons (TDC) is a platform that provides AI-ready datasets and learning tasks for drug discovery and development. It offers a comprehensive ecosystem of tools, libraries, leaderboards, and community resources for therapeutics machine learning according to the TDC website.

TDC datasets cover a wide range of therapeutic products (small molecules, biologics, gene editing therapies) across the entire drug development pipeline (target identification, hit discovery, lead optimization, and manufacturing). These datasets are ML-ready, meaning input features are processed into an accessible format that scientists can use directly as input to machine learning methods.

## Features

This script enables you to:

1. **List Available Datasets**: View all datasets available in TDC organized by problem type and task.
2. **Download Datasets**: Easily download any dataset with proper train/validation/test splits.
3. **Export Standardized CSVs**: Export datasets in a standardized format where:
   - The molecule column is always named "SMILES" 
   - The data/target column is named after the dataset
   - Files are saved in a consistent, standardized format
4. **Prepare for ML**: Process datasets to make them ready for machine learning, including:
   - Handling missing values
   - Scaling numerical features
   - Encoding categorical variables
   - Creating train/validation/test splits
   - Generating data summaries and visualizations

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/tdc-dataset-manager.git
cd tdc-dataset-manager
```

2. Install the required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm PyTDC
```

The script will also automatically install the TDC package if it's not already installed.

## Usage

### List Available Datasets

```bash
# List all available datasets
python tdc_ml_script.py --list

# List datasets for a specific problem type
python tdc_ml_script.py --list --problem-type single_pred

# List datasets for a specific task
python tdc_ml_script.py --list --problem-type single_pred --task-name ADME
```

### Download Datasets

```bash
# Download a dataset
python tdc_ml_script.py --download --problem-type single_pred --task-name ADME --dataset-name Caco2_Wang

# Download with a specific split method
python tdc_ml_script.py --download --problem-type single_pred --task-name ADME --dataset-name Caco2_Wang --split-method scaffold
```

### Export Standardized CSVs

```bash
# Export a dataset as a standardized CSV
python tdc_ml_script.py --export-csv --problem-type single_pred --task-name ADME --dataset-name Caco2_Wang
```

### Prepare for Machine Learning

```bash
# Prepare a dataset for machine learning
python tdc_ml_script.py --prepare --problem-type single_pred --task-name ADME --dataset-name Caco2_Wang

# Prepare with custom options
python tdc_ml_script.py --prepare --problem-type single_pred --task-name ADME --dataset-name Caco2_Wang --split-method scaffold --feature-type morgan --split-ratio 0.7,0.15,0.15 --data-dir custom_data_dir
```

## Standardized CSV Format

When using the `--export-csv` option, the script exports datasets in a consistent format that's ideal for machine learning:

1. **SMILES Column**: All molecular structures are placed in a column named "SMILES", regardless of the original column name in TDC.

2. **Dataset-Named Target Column**: The property/activity values are placed in a column named after the dataset itself.

For example, if you export the "Caco2_Wang" dataset, the resulting CSV will have:
- A "SMILES" column containing molecular structures
- A "Caco2_Wang" column containing the permeability values

For more complex datasets like drug-target interactions (DTI):
- "SMILES" column for the drug molecules
- "Target" column for the protein/target identifiers
- Dataset-named column for the interaction values

This standardized format makes it easier to:
- Process multiple datasets with consistent column names
- Integrate datasets into ML pipelines
- Compare results across different datasets

## Command Line Arguments

- `--list`: List available datasets
- `--download`: Download a dataset
- `--prepare`: Prepare dataset for ML
- `--export-csv`: Export dataset as a standardized CSV (SMILES column + dataset-named target column)
- `--problem-type`: Filter by problem type (`single_pred`, `multi_pred`, `generation`)
- `--task-name`: Specify the task name (e.g., ADME, Tox, DTI)
- `--dataset-name`: Name of the dataset to download/prepare/export
- `--split-method`: Data split method (random, scaffold, etc.)
- `--no-preprocess`: Skip preprocessing steps when preparing for ML
- `--feature-type`: Feature type to use (default, morgan, etc.)
- `--split-ratio`: Custom train/val/test split ratio (comma-separated, e.g., 0.8,0.1,0.1)
- `--data-dir`: Directory to store downloaded datasets

## TDC Compatibility

This script is designed to work with different versions of the TDC package. It will automatically detect which task modules are available in your particular TDC installation.

Based on the TDC documentation, the following modules should be available:

### Single Prediction Tasks
- ADME: Absorption, Distribution, Metabolism, Excretion properties
- Tox: Toxicity prediction
- HTS: High-Throughput Screening
- CRISPROutcome: CRISPR guide efficiency prediction
- Epitope: Epitope prediction
- Paratope: Paratope prediction
- QM: Quantum Mechanics
- Develop: Development and manufacturing

### Multi Prediction Tasks
- DTI: Drug-Target Interaction
- DDI: Drug-Drug Interaction
- PPI: Protein-Protein Interaction
- GDA: Gene-Disease Association
- DrugSyn: Drug Synergy
- PeptideMHC: Peptide-MHC binding

### Generation Tasks
- RetroSyn: Retrosynthesis
- Reaction: Reaction prediction
- MolGen: Molecule generation
- PepGen: Peptide generation
- ABGen: Antibody generation

If you encounter import errors, they are likely due to differences in your TDC version. The script will automatically adapt to the available modules in your installation.

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- Adding support for new TDC modules and datasets
- Improving preprocessing options
- Enhancing visualization capabilities
- Adding new export formats for different ML frameworks
- Documentation improvements
- Bug fixes

## Acknowledgments

- [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) for providing the datasets and API
- [TDC GitHub Repository](https://github.com/mims-harvard/TDC)
- The TDC team for their work in organizing therapeutics data

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** While this tool is MIT licensed, the datasets from TDC may have their own licenses and terms of use. Please refer to the TDC documentation for information about the license terms for individual datasets.