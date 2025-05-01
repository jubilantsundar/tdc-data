# Therapeutics Data Commons (TDC) Dataset Manager

This script helps you explore, download, and prepare datasets from the Therapeutics Data Commons (TDC) for machine learning tasks.

## About Therapeutics Data Commons

The Therapeutics Data Commons (TDC) is a platform that provides AI-ready datasets and learning tasks for drug discovery and development. It offers a comprehensive ecosystem of tools, libraries, leaderboards, and community resources for therapeutics machine learning.

TDC datasets cover a wide range of therapeutic products (small molecules, biologics, gene editing therapies) across the entire drug development pipeline (target identification, hit discovery, lead optimization, and manufacturing). These datasets are ML-ready, meaning input features are processed into an accessible format that scientists can use directly as input to machine learning methods.

## Features of This Script

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

The script will automatically install the TDC package if it's not already installed. You'll need Python 3.7+ and pip.

Additional dependencies that will be installed:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm

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

## Usage

### Basic Commands

```bash
# List all available datasets
python tdc_ml_script.py --list

# List datasets for a specific problem type
python tdc_ml_script.py --list --problem-type single_pred

# List datasets for a specific task
python tdc_ml_script.py --list --problem-type single_pred --task-name ADME

# Download a dataset
python tdc_ml_script.py --download --problem-type single_pred --task-name ADME --dataset-name Caco2_Wang

# Prepare a dataset for machine learning
python tdc_ml_script.py --prepare --problem-type single_pred --task-name ADME --dataset-name Caco2_Wang

# Export a dataset as a standardized CSV
python tdc_ml_script.py --export-csv --problem-type single_pred --task-name ADME --dataset-name Caco2_Wang
```

### Advanced Options

```bash
# Download with a specific split method
python tdc_ml_script.py --download --problem-type single_pred --task-name ADME --dataset-name Caco2_Wang --split-method scaffold

# Prepare with custom options
python tdc_ml_script.py --prepare --problem-type single_pred --task-name ADME --dataset-name Caco2_Wang --split-method scaffold --feature-type morgan --split-ratio 0.7,0.15,0.15 --data-dir custom_data_dir
```

### Command Line Arguments

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