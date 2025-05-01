#!/usr/bin/env python
# Therapeutics Data Commons (TDC) Dataset Manager
# This script allows users to:
# 1. List available datasets from TDC
# 2. Download selected datasets
# 3. Prepare datasets for machine learning

import os
import sys
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('TDC-ML-Manager')

# Check if TDC is installed, if not install it
try:
    import tdc
    from tdc.utils import retrieve_dataset_names
    logger.info("TDC package already installed.")
    
    # Check TDC version
    import pkg_resources
    tdc_version = pkg_resources.get_distribution("PyTDC").version
    logger.info(f"TDC version: {tdc_version}")
except ImportError:
    logger.info("TDC package not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyTDC"])
    import tdc
    from tdc.utils import retrieve_dataset_names
    logger.info("TDC package installed successfully.")
    
    # Get TDC version after installation
    import pkg_resources
    tdc_version = pkg_resources.get_distribution("PyTDC").version
    logger.info(f"TDC version: {tdc_version}")

class TDCDatasetManager:
    """Class to manage TDC datasets for machine learning preparation"""
    
    def __init__(self, data_dir="tdc_data"):
        """Initialize the TDC dataset manager
        
        Args:
            data_dir (str): Directory to store downloaded datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Data will be stored in: {os.path.abspath(data_dir)}")
        
        # Base map of problem types and their corresponding classes in TDC
        self.problem_types = {
            'single_pred': ['ADME', 'Tox', 'HTS', 'CRISPROutcome', 'Epitope', 
                           'Paratope', 'QM', 'Develop'],
            'multi_pred': ['DTI', 'DDI', 'PPI', 'GDA', 'DrugSyn', 'PeptideMHC', 'AntibodyAff', 'Catalyst'],
            'generation': ['RetroSyn', 'Reaction', 'MolGen', 'PepGen', 'ABGen']
        }
        
        # Verify which modules are actually available
        self._verify_available_modules()
    
    def _verify_available_modules(self):
        """Verify which TDC modules are actually available in the installation"""
        # For each problem type
        for problem_type, tasks in self.problem_types.items():
            available_tasks = []
            
            # Try to import each task module individually
            for task in tasks:
                try:
                    if problem_type == 'single_pred':
                        # Import individually to avoid errors if one module is missing
                        if task == 'ADME':
                            from tdc.single_pred import ADME
                            available_tasks.append(task)
                        elif task == 'Tox':
                            from tdc.single_pred import Tox
                            available_tasks.append(task)
                        elif task == 'HTS':
                            from tdc.single_pred import HTS
                            available_tasks.append(task)
                        elif task == 'CRISPROutcome':
                            from tdc.single_pred import CRISPROutcome
                            available_tasks.append(task)
                        elif task == 'Epitope':
                            from tdc.single_pred import Epitope
                            available_tasks.append(task)
                        elif task == 'Paratope':
                            from tdc.single_pred import Paratope
                            available_tasks.append(task)
                        elif task == 'QM':
                            from tdc.single_pred import QM
                            available_tasks.append(task)
                        elif task == 'Develop':
                            from tdc.single_pred import Develop
                            available_tasks.append(task)
                    
                    elif problem_type == 'multi_pred':
                        if task == 'DTI':
                            from tdc.multi_pred import DTI
                            available_tasks.append(task)
                        elif task == 'DDI':
                            from tdc.multi_pred import DDI
                            available_tasks.append(task)
                        elif task == 'PPI':
                            from tdc.multi_pred import PPI
                            available_tasks.append(task)
                        elif task == 'GDA':
                            from tdc.multi_pred import GDA
                            available_tasks.append(task)
                        elif task == 'DrugSyn':
                            from tdc.multi_pred import DrugSyn
                            available_tasks.append(task)
                        elif task == 'PeptideMHC':
                            from tdc.multi_pred import PeptideMHC
                            available_tasks.append(task)
                        elif task == 'AntibodyAff':
                            from tdc.multi_pred import AntibodyAff
                            available_tasks.append(task)
                        elif task == 'Catalyst':
                            from tdc.multi_pred import Catalyst
                            available_tasks.append(task)
                    
                    elif problem_type == 'generation':
                        if task == 'RetroSyn':
                            from tdc.generation import RetroSyn
                            available_tasks.append(task)
                        elif task == 'Reaction':
                            from tdc.generation import Reaction
                            available_tasks.append(task)
                        elif task == 'MolGen':
                            from tdc.generation import MolGen
                            available_tasks.append(task)
                        elif task == 'PepGen':
                            from tdc.generation import PepGen
                            available_tasks.append(task)
                        elif task == 'ABGen':
                            from tdc.generation import ABGen
                            available_tasks.append(task)
                
                except ImportError as e:
                    logger.warning(f"Task module '{task}' from '{problem_type}' is not available: {str(e)}")
                    continue
                except Exception as e:
                    logger.warning(f"Error checking task '{task}' from '{problem_type}': {str(e)}")
                    continue
            
            # Update the problem_types dictionary with only available tasks
            self.problem_types[problem_type] = available_tasks
            
        # Log available tasks
        logger.info("Available TDC modules:")
        for problem_type, tasks in self.problem_types.items():
            logger.info(f"  {problem_type}: {', '.join(tasks)}")
    
    def list_available_datasets(self, problem_type=None, task_name=None):
        """List available datasets from TDC
        
        Args:
            problem_type (str, optional): Filter by problem type 
                (single_pred, multi_pred, generation)
            task_name (str, optional): Filter by specific task within a problem type
                
        Returns:
            dict: Dictionary of available datasets organized by problem type and task
        """
        available_datasets = {}
        
        # If problem_type is specified, only show datasets for that problem type
        if problem_type:
            if problem_type not in self.problem_types:
                raise ValueError(f"Invalid problem type. Choose from: {list(self.problem_types.keys())}")
            
            problem_types_to_check = [problem_type]
        else:
            problem_types_to_check = self.problem_types.keys()
        
        # For each problem type
        for prob_type in problem_types_to_check:
            available_datasets[prob_type] = {}
            
            # If task_name is specified, only show datasets for that task
            if task_name and prob_type == problem_type:
                if task_name not in self.problem_types[prob_type]:
                    raise ValueError(f"Invalid task name for {prob_type}. Choose from: {self.problem_types[prob_type]}")
                tasks_to_check = [task_name]
            else:
                tasks_to_check = self.problem_types[prob_type]
            
            # For each task in the problem type
            for task in tasks_to_check:
                try:
                    datasets = retrieve_dataset_names(task)
                    available_datasets[prob_type][task] = datasets
                except Exception as e:
                    logger.warning(f"Could not retrieve datasets for task: {task}. Error: {str(e)}")
                    available_datasets[prob_type][task] = []
        
        return available_datasets
    
    def display_available_datasets(self, problem_type=None, task_name=None):
        """Display available datasets in a user-friendly format
        
        Args:
            problem_type (str, optional): Filter by problem type
            task_name (str, optional): Filter by specific task
        """
        available_datasets = self.list_available_datasets(problem_type, task_name)
        
        print("\n" + "="*80)
        print(f"{'AVAILABLE DATASETS FROM THERAPEUTICS DATA COMMONS':^80}")
        print("="*80)
        
        for prob_type, tasks in available_datasets.items():
            print(f"\n{prob_type.upper()} PROBLEMS:")
            print("-" * 40)
            
            for task, datasets in tasks.items():
                if datasets:
                    print(f"\n  Task: {task} ({len(datasets)} datasets)")
                    # Print datasets in columns
                    max_name_length = max([len(name) for name in datasets]) if datasets else 0
                    col_width = max(max_name_length + 2, 25)
                    num_cols = max(1, 80 // col_width)
                    
                    for i in range(0, len(datasets), num_cols):
                        row_datasets = datasets[i:i+num_cols]
                        print("    " + "".join(name.ljust(col_width) for name in row_datasets))
        
        print("\n" + "="*80)
    
    def download_dataset(self, problem_type, task_name, dataset_name, split_method=None):
        """Download a specific dataset from TDC
        
        Args:
            problem_type (str): Problem type (single_pred, multi_pred, generation)
            task_name (str): Task name
            dataset_name (str): Dataset name
            split_method (str, optional): Data split method 
                (random, scaffold, etc.)
                
        Returns:
            tuple: (data, splits) where data is the complete dataset and 
                  splits contains train/val/test splits if applicable
        """
        if problem_type not in self.problem_types:
            raise ValueError(f"Invalid problem type. Choose from: {list(self.problem_types.keys())}")
        
        if task_name not in self.problem_types[problem_type]:
            raise ValueError(f"Invalid task name. Choose from: {self.problem_types[problem_type]}")
        
        # Import the right module based on problem type
        if problem_type == 'single_pred':
            task_class = None
            
            try:
                if task_name == 'ADME':
                    from tdc.single_pred import ADME
                    task_class = ADME
                elif task_name == 'Tox':
                    from tdc.single_pred import Tox
                    task_class = Tox
                elif task_name == 'HTS':
                    from tdc.single_pred import HTS
                    task_class = HTS
                elif task_name == 'CRISPROutcome':
                    from tdc.single_pred import CRISPROutcome
                    task_class = CRISPROutcome
                elif task_name == 'Epitope':
                    from tdc.single_pred import Epitope
                    task_class = Epitope
                elif task_name == 'Paratope':
                    from tdc.single_pred import Paratope
                    task_class = Paratope
                elif task_name == 'QM':
                    from tdc.single_pred import QM
                    task_class = QM
                elif task_name == 'Develop':
                    from tdc.single_pred import Develop
                    task_class = Develop
                else:
                    raise ValueError(f"Invalid task name '{task_name}' for problem type '{problem_type}'")
            except ImportError as e:
                raise ImportError(f"Could not import {task_name} from tdc.single_pred: {str(e)}")
            
        elif problem_type == 'multi_pred':
            task_class = None
            
            try:
                if task_name == 'DTI':
                    from tdc.multi_pred import DTI
                    task_class = DTI
                elif task_name == 'DDI':
                    from tdc.multi_pred import DDI
                    task_class = DDI
                elif task_name == 'PPI':
                    from tdc.multi_pred import PPI
                    task_class = PPI
                elif task_name == 'GDA':
                    from tdc.multi_pred import GDA
                    task_class = GDA
                elif task_name == 'DrugSyn':
                    from tdc.multi_pred import DrugSyn
                    task_class = DrugSyn
                elif task_name == 'PeptideMHC':
                    from tdc.multi_pred import PeptideMHC
                    task_class = PeptideMHC
                elif task_name == 'AntibodyAff':
                    from tdc.multi_pred import AntibodyAff
                    task_class = AntibodyAff
                elif task_name == 'Catalyst':
                    from tdc.multi_pred import Catalyst
                    task_class = Catalyst
                else:
                    raise ValueError(f"Invalid task name '{task_name}' for problem type '{problem_type}'")
            except ImportError as e:
                raise ImportError(f"Could not import {task_name} from tdc.multi_pred: {str(e)}")
            
        elif problem_type == 'generation':
            task_class = None
            
            try:
                if task_name == 'RetroSyn':
                    from tdc.generation import RetroSyn
                    task_class = RetroSyn
                elif task_name == 'Reaction':
                    from tdc.generation import Reaction
                    task_class = Reaction
                elif task_name == 'MolGen':
                    from tdc.generation import MolGen
                    task_class = MolGen
                elif task_name == 'PepGen':
                    from tdc.generation import PepGen
                    task_class = PepGen
                elif task_name == 'ABGen':
                    from tdc.generation import ABGen
                    task_class = ABGen
                else:
                    raise ValueError(f"Invalid task name '{task_name}' for problem type '{problem_type}'")
            except ImportError as e:
                raise ImportError(f"Could not import {task_name} from tdc.generation: {str(e)}")
        
        else:
            raise ValueError(f"Invalid problem type '{problem_type}'. Choose from: {list(self.problem_types.keys())}")
        
        if task_class is None:
            raise ImportError(f"Could not import task class for {task_name}")
        
        logger.info(f"Downloading {dataset_name} from {task_name} ({problem_type})...")
        
        logger.info(f"Downloading {dataset_name} from {task_name} ({problem_type})...")
        
        # Create data object
        try:
            data = task_class(name=dataset_name)
            logger.info(f"Dataset '{dataset_name}' loaded successfully!")
            
            # Save complete dataset
            os.makedirs(os.path.join(self.data_dir, problem_type, task_name), exist_ok=True)
            data_df = data.get_data(format='df')
            
            # Standardize column names for molecule data
            # Many TDC datasets have 'Drug', 'SMILES', or 'compound_iso_smiles' as the molecule column
            # Rename to ensure SMILES is the consistent column name for molecular structures
            mol_column_mapping = {
                'Drug': 'SMILES',
                'drug': 'SMILES',
                'compound_iso_smiles': 'SMILES',
                'smiles': 'SMILES',
                'canonical_smiles': 'SMILES',
                'molecule': 'SMILES'
            }
            
            # Rename molecule column if present
            for old_col, new_col in mol_column_mapping.items():
                if old_col in data_df.columns:
                    data_df = data_df.rename(columns={old_col: new_col})
                    logger.info(f"Renamed column '{old_col}' to '{new_col}'")
            
            # Rename target column to match dataset name if Y or y is present
            if 'Y' in data_df.columns:
                data_df = data_df.rename(columns={'Y': dataset_name})
                logger.info(f"Renamed column 'Y' to '{dataset_name}'")
            elif 'y' in data_df.columns:
                data_df = data_df.rename(columns={'y': dataset_name})
                logger.info(f"Renamed column 'y' to '{dataset_name}'")
            
            data_path = os.path.join(self.data_dir, problem_type, task_name, f"{dataset_name}.csv")
            data_df.to_csv(data_path, index=False)
            logger.info(f"Complete dataset saved to {data_path}")
            
            # Get splits if applicable and if split_method is provided
            splits = None
            if split_method and hasattr(data, 'get_split'):
                logger.info(f"Creating dataset splits using '{split_method}' method...")
                try:
                    splits = data.get_split(method=split_method)
                    
                    # Save splits
                    splits_dir = os.path.join(self.data_dir, problem_type, task_name, f"{dataset_name}_splits")
                    os.makedirs(splits_dir, exist_ok=True)
                    
                    for split_name, split_data in splits.items():
                        split_path = os.path.join(splits_dir, f"{split_name}.csv")
                        if isinstance(split_data, pd.DataFrame):
                            split_df = split_data.copy()
                            
                            # Apply the same renaming to the splits
                            # Rename molecule column if present
                            mol_column_mapping = {
                                'Drug': 'SMILES',
                                'drug': 'SMILES',
                                'compound_iso_smiles': 'SMILES',
                                'smiles': 'SMILES',
                                'canonical_smiles': 'SMILES',
                                'molecule': 'SMILES'
                            }
                            
                            for old_col, new_col in mol_column_mapping.items():
                                if old_col in split_df.columns:
                                    split_df = split_df.rename(columns={old_col: new_col})
                            
                            # Rename target column to match dataset name if Y or y is present
                            if 'Y' in split_df.columns:
                                split_df = split_df.rename(columns={'Y': dataset_name})
                            elif 'y' in split_df.columns:
                                split_df = split_df.rename(columns={'y': dataset_name})
                            
                            split_df.to_csv(split_path, index=False)
                        else:
                            # Convert to DataFrame if it's not already
                            split_df = pd.DataFrame(split_data)
                            split_df.to_csv(split_path, index=False)
                    
                    logger.info(f"Dataset splits saved to {splits_dir}")
                except Exception as e:
                    logger.warning(f"Could not create splits: {str(e)}")
                    splits = None
            
            return data, splits
        
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            return None, None
    
    def prepare_for_ml(self, problem_type, task_name, dataset_name, split_method=None, 
                       preprocess=True, feature_type='default', split_ratio=None):
        """Prepare dataset for machine learning
        
        Args:
            problem_type (str): Problem type
            task_name (str): Task name
            dataset_name (str): Dataset name
            split_method (str, optional): Data split method
            preprocess (bool): Whether to preprocess the data
            feature_type (str): Feature type to use ('default', 'morgan', etc.)
            split_ratio (tuple): Custom train/val/test split ratio (e.g., (0.7, 0.1, 0.2))
            
        Returns:
            dict: Dictionary containing prepared data for ML
        """
        # First download the dataset
        data, splits = self.download_dataset(problem_type, task_name, dataset_name, split_method)
        
        if data is None:
            return None
        
        # Prepare output dictionary
        ml_data = {
            'dataset_info': {
                'name': dataset_name,
                'problem_type': problem_type,
                'task_name': task_name
            },
            'data': data.get_data(format='df'),
            'splits': {}
        }
        
        # Process splits
        if splits:
            logger.info("Using TDC provided splits")
            ml_data['splits'] = splits
        elif split_ratio:
            logger.info(f"Creating custom splits with ratio {split_ratio}")
            df = data.get_data(format='df')
            
            # Extract features and labels
            if 'Y' in df.columns:
                y_col = 'Y'
            elif 'y' in df.columns:
                y_col = 'y'
            else:
                # Try to guess the label column
                possible_y_cols = ['label', 'target', 'activity', 'value']
                for col in possible_y_cols:
                    if col in df.columns:
                        y_col = col
                        break
                else:
                    logger.warning("Could not identify label column. Using last column as label.")
                    y_col = df.columns[-1]
            
            X = df.drop(y_col, axis=1)
            y = df[y_col]
            
            # Create train/validation/test splits
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=split_ratio[2], random_state=42
            )
            
            # Adjust validation split ratio
            val_ratio = split_ratio[1] / (split_ratio[0] + split_ratio[1])
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio, random_state=42
            )
            
            # Store splits
            ml_data['splits'] = {
                'train': pd.concat([X_train, y_train], axis=1),
                'valid': pd.concat([X_val, y_val], axis=1),
                'test': pd.concat([X_test, y_test], axis=1)
            }
        
        # Preprocess data if requested
        if preprocess:
            logger.info("Preprocessing data...")
            # Extract features and labels from each split
            processed_splits = {}
            
            for split_name, split_data in ml_data['splits'].items():
                if isinstance(split_data, pd.DataFrame):
                    # Identify feature and label columns
                    label_cols = []
                    for col in ['Y', 'y', 'label', 'target', 'activity', 'value']:
                        if col in split_data.columns:
                            label_cols.append(col)
                    
                    if not label_cols:
                        logger.warning(f"No label column found in {split_name} split. Skipping preprocessing.")
                        processed_splits[split_name] = split_data
                        continue
                    
                    # Extract features and labels
                    y = split_data[label_cols]
                    X = split_data.drop(label_cols, axis=1)
                    
                    # Handle text, categorical, and numerical features
                    X_processed = self._preprocess_features(X, is_train=(split_name=='train'))
                    
                    # Combine processed features and labels
                    processed_splits[split_name] = pd.concat([X_processed, y], axis=1)
                else:
                    logger.warning(f"Split '{split_name}' is not a DataFrame. Skipping preprocessing.")
                    processed_splits[split_name] = split_data
            
            ml_data['preprocessed_splits'] = processed_splits
        
        # If feature type is specified and different from default, try to get alternative features
        if feature_type != 'default' and hasattr(data, 'get_' + feature_type + '_features'):
            try:
                logger.info(f"Extracting {feature_type} features...")
                feature_method = getattr(data, 'get_' + feature_type + '_features')
                ml_data['alternative_features'] = feature_method()
            except Exception as e:
                logger.warning(f"Could not extract {feature_type} features: {str(e)}")
        
        # Save prepared data
        output_dir = os.path.join(self.data_dir, problem_type, task_name, f"{dataset_name}_ml_ready")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "ml_data.pickle"), 'wb') as f:
            pickle.dump(ml_data, f)
        
        logger.info(f"ML-ready data saved to {output_dir}")
        
        # Create a quick data summary
        self._create_data_summary(ml_data, output_dir)
        
        return ml_data
    
    def _preprocess_features(self, X, is_train=True):
        """Preprocess features for ML
        
        Args:
            X (DataFrame): Features to preprocess
            is_train (bool): Whether this is the training set
            
        Returns:
            DataFrame: Preprocessed features
        """
        # Make a copy to avoid modifying the original
        X_processed = X.copy()
        
        # Simple preprocessing for different column types
        numeric_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        
        # Handle missing values
        for col in X_processed.columns:
            if X_processed[col].isna().any():
                if col in numeric_cols:
                    X_processed[col] = X_processed[col].fillna(X_processed[col].mean())
                else:
                    X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0])
        
        # Scale numerical features
        if len(numeric_cols) > 0:
            if is_train:
                self.scaler = StandardScaler()
                X_processed[numeric_cols] = self.scaler.fit_transform(X_processed[numeric_cols])
            else:
                # Make sure scaler exists
                if hasattr(self, 'scaler'):
                    X_processed[numeric_cols] = self.scaler.transform(X_processed[numeric_cols])
                else:
                    logger.warning("Scaler not found. Using untransformed numeric features.")
        
        # Handle categorical features with one-hot encoding
        if len(categorical_cols) > 0:
            X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
        
        return X_processed
    
    def _create_data_summary(self, ml_data, output_dir):
        """Create a summary of the prepared data
        
        Args:
            ml_data (dict): ML-ready data
            output_dir (str): Directory to save summary
        """
        # Get basic dataset info
        dataset_name = ml_data['dataset_info']['name']
        problem_type = ml_data['dataset_info']['problem_type']
        task_name = ml_data['dataset_info']['task_name']
        
        with open(os.path.join(output_dir, "dataset_summary.txt"), 'w') as f:
            f.write(f"DATASET SUMMARY: {dataset_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Problem Type: {problem_type}\n")
            f.write(f"Task: {task_name}\n\n")
            
            # Overall data stats
            full_data = ml_data['data']
            f.write(f"Total samples: {len(full_data)}\n")
            f.write(f"Features: {full_data.shape[1]}\n\n")
            
            # Split info
            f.write("DATA SPLITS:\n")
            f.write("-"*40 + "\n")
            for split_name, split_data in ml_data['splits'].items():
                if isinstance(split_data, pd.DataFrame):
                    f.write(f"{split_name}: {len(split_data)} samples\n")
            
            # Feature information
            f.write("\nFEATURE INFORMATION:\n")
            f.write("-"*40 + "\n")
            for col in full_data.columns:
                col_type = full_data[col].dtype
                missing = full_data[col].isna().sum()
                if pd.api.types.is_numeric_dtype(full_data[col]):
                    f.write(f"{col} ({col_type}): min={full_data[col].min():.4f}, "
                           f"max={full_data[col].max():.4f}, "
                           f"mean={full_data[col].mean():.4f}, "
                           f"missing={missing}\n")
                else:
                    unique_vals = full_data[col].nunique()
                    f.write(f"{col} ({col_type}): {unique_vals} unique values, "
                           f"missing={missing}\n")
        
        logger.info(f"Dataset summary saved to {os.path.join(output_dir, 'dataset_summary.txt')}")
        
        # Create some basic visualizations if possible
        try:
            self._create_visualizations(ml_data, output_dir)
        except Exception as e:
            logger.warning(f"Could not create visualizations: {str(e)}")
    
    def export_standardized_csv(self, problem_type, task_name, dataset_name):
        """Export dataset as a standardized CSV with SMILES and target columns
        
        This function specifically exports the dataset with the molecule column as SMILES
        and the target column named after the dataset.
        
        Args:
            problem_type (str): Problem type (single_pred, multi_pred, generation)
            task_name (str): Task name
            dataset_name (str): Dataset name
            
        Returns:
            str: Path to the exported CSV file
        """
        # First check if dataset is already downloaded
        dataset_path = os.path.join(self.data_dir, problem_type, task_name, f"{dataset_name}.csv")
        
        if os.path.exists(dataset_path):
            logger.info(f"Dataset already exists at {dataset_path}, using existing file.")
            data_df = pd.read_csv(dataset_path)
        else:
            # Download the dataset
            data, _ = self.download_dataset(problem_type, task_name, dataset_name)
            if data is None:
                return None
            data_df = data.get_data(format='df')
        
        # Create standardized directory
        std_dir = os.path.join(self.data_dir, "standardized")
        os.makedirs(std_dir, exist_ok=True)
        
        # Identify and rename columns
        # This section helps handle different TDC dataset formats
        
        # 1. Find SMILES column - many TDC datasets use different column names for molecular structures
        smiles_column = None
        possible_smiles_cols = ['SMILES', 'Drug', 'drug', 'compound_iso_smiles', 'smiles', 
                               'canonical_smiles', 'molecule', 'Ligand']
        
        for col in possible_smiles_cols:
            if col in data_df.columns:
                smiles_column = col
                break
        
        if smiles_column is None:
            logger.warning(f"Could not identify SMILES column in {dataset_name}. This may not be a molecular dataset.")
            # For non-molecular datasets, just save as is
            std_path = os.path.join(std_dir, f"{dataset_name}_standard.csv")
            data_df.to_csv(std_path, index=False)
            return std_path
        
        # 2. Find target column (if not a molecular dataset, this might not exist)
        target_column = None
        possible_target_cols = ['Y', 'y', 'label', 'target', 'activity', 'value', 'property', 'outcome']
        
        for col in possible_target_cols:
            if col in data_df.columns:
                target_column = col
                break
        
        # Prepare standardized DataFrame
        std_df = pd.DataFrame()
        
        # Always include SMILES column
        std_df['SMILES'] = data_df[smiles_column]
        
        # Include target column if found, named after the dataset
        if target_column:
            std_df[dataset_name] = data_df[target_column]
        
        # For paired datasets (like DTI), include the target entity
        target_entity_col = None
        possible_target_entity_cols = ['Target', 'target', 'Protein', 'protein', 'Target_ID']
        
        for col in possible_target_entity_cols:
            if col in data_df.columns:
                target_entity_col = col
                break
        
        if target_entity_col:
            std_df['Target'] = data_df[target_entity_col]
        
        # Save standardized CSV
        std_path = os.path.join(std_dir, f"{dataset_name}_standard.csv")
        std_df.to_csv(std_path, index=False)
        logger.info(f"Standardized dataset saved to {std_path}")
        
        return std_path
    
    def _create_visualizations(self, ml_data, output_dir):
        """Create basic visualizations of the data
        
        Args:
            ml_data (dict): ML-ready data
            output_dir (str): Directory to save visualizations
        """
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        full_data = ml_data['data']
        
        # Identify label column
        label_cols = []
        for col in ['Y', 'y', 'label', 'target', 'activity', 'value']:
            if col in full_data.columns:
                label_cols.append(col)
        
        if not label_cols:
            logger.warning("No label column found. Skipping label-based visualizations.")
            return
        
        label_col = label_cols[0]
        
        # Distribution of the target variable
        plt.figure(figsize=(10, 6))
        if pd.api.types.is_numeric_dtype(full_data[label_col]):
            sns.histplot(full_data[label_col].dropna(), kde=True)
            plt.title(f'Distribution of {label_col}')
        else:
            sns.countplot(y=full_data[label_col].dropna())
            plt.title(f'Count of {label_col} Categories')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "target_distribution.png"))
        plt.close()
        
        # Correlation matrix for numerical features
        numeric_cols = full_data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1 and len(numeric_cols) <= 20:  # Only if there's a reasonable number
            plt.figure(figsize=(12, 10))
            corr_matrix = full_data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "correlation_matrix.png"))
            plt.close()
        
        logger.info(f"Visualizations saved to {viz_dir}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='TDC Dataset Manager for Machine Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main actions
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--download', action='store_true', help='Download a dataset')
    parser.add_argument('--prepare', action='store_true', help='Prepare dataset for ML')
    parser.add_argument('--export-csv', action='store_true', 
                       help='Export dataset as standardized CSV with SMILES and target columns')
    
    # Filtering options for listing
    parser.add_argument('--problem-type', type=str, choices=['single_pred', 'multi_pred', 'generation'],
                       help='Filter by problem type')
    parser.add_argument('--task-name', type=str, help='Filter by task name')
    
    # Dataset selection
    parser.add_argument('--dataset-name', type=str, help='Name of the dataset to download/prepare')
    
    # Preparation options
    parser.add_argument('--split-method', type=str, default='scaffold',
                       help='Data split method (random, scaffold, etc.)')
    parser.add_argument('--no-preprocess', action='store_true', help='Skip preprocessing')
    parser.add_argument('--feature-type', type=str, default='default',
                       help='Feature type to use (default, morgan, etc.)')
    parser.add_argument('--split-ratio', type=str, default='0.8,0.1,0.1',
                       help='Custom train/val/test split ratio (comma-separated)')
    
    # Output directory
    parser.add_argument('--data-dir', type=str, default='tdc_data',
                       help='Directory to store downloaded datasets')
    
    args = parser.parse_args()
    
    # Convert split ratio to tuple
    if args.split_ratio:
        try:
            args.split_ratio = tuple(map(float, args.split_ratio.split(',')))
            assert len(args.split_ratio) == 3
            assert sum(args.split_ratio) == 1.0
        except:
            parser.error("Split ratio must be three comma-separated numbers that sum to 1.0")
    
    # Validate that at least one action is specified
    if not (args.list or args.download or args.prepare or args.export_csv):
        parser.error("At least one action (--list, --download, --prepare, or --export-csv) must be specified")
    
    # Validate that required arguments are provided for download, prepare, and export-csv
    if args.download or args.prepare or args.export_csv:
        if not args.problem_type:
            parser.error("--problem-type is required for download, prepare, and export-csv actions")
        if not args.task_name:
            parser.error("--task-name is required for download, prepare, and export-csv actions")
        if not args.dataset_name:
            parser.error("--dataset-name is required for download, prepare, and export-csv actions")
    
    return args

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize dataset manager
    manager = TDCDatasetManager(data_dir=args.data_dir)
    
    # Execute requested actions
    if args.list:
        manager.display_available_datasets(args.problem_type, args.task_name)
    
    if args.download:
        data, splits = manager.download_dataset(
            args.problem_type, args.task_name, args.dataset_name, args.split_method
        )
        
        if data is not None:
            print(f"\nSuccessfully downloaded dataset: {args.dataset_name}")
            print(f"Data shape: {data.get_data(format='df').shape}")
            
            if splits:
                print("\nData splits:")
                for split_name, split_data in splits.items():
                    if isinstance(split_data, pd.DataFrame):
                        print(f"  {split_name}: {split_data.shape[0]} samples")
                    else:
                        print(f"  {split_name}: {len(split_data)} samples")
    
    if args.prepare:
        ml_data = manager.prepare_for_ml(
            args.problem_type, args.task_name, args.dataset_name,
            args.split_method, not args.no_preprocess, args.feature_type, args.split_ratio
        )
        
        if ml_data is not None:
            print(f"\nSuccessfully prepared dataset for ML: {args.dataset_name}")
            print(f"Output directory: {os.path.join(args.data_dir, args.problem_type, args.task_name, args.dataset_name + '_ml_ready')}")
    
    if args.export_csv:
        csv_path = manager.export_standardized_csv(
            args.problem_type, args.task_name, args.dataset_name
        )
        
        if csv_path is not None:
            print(f"\nSuccessfully exported standardized CSV for: {args.dataset_name}")
            print(f"Standard CSV file: {csv_path}")
            
            # Print a preview of the CSV
            try:
                df = pd.read_csv(csv_path)
                print("\nPreview of standardized CSV file:")
                print("-" * 80)
                print(df.head(5).to_string())
                print("-" * 80)
                print(f"Total rows: {len(df)}, Total columns: {df.shape[1]}")
                
                # Print column information
                print("\nColumn information:")
                for col in df.columns:
                    if col == 'SMILES':
                        print(f"- SMILES: Contains molecular structures in SMILES format")
                    elif col == 'Target':
                        print(f"- Target: Contains target entities (proteins, genes, etc.)")
                    else:
                        print(f"- {col}: Contains target values/properties for the dataset")
            except Exception as e:
                print(f"Could not display preview: {str(e)}")

if __name__ == "__main__":
    main()
