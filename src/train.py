"""
Main training script for the project.
"""

# Import basic libraries
import os
import sys
import json
from datetime import datetime
from datasets import load_dataset
from pathlib import Path

# Load the DistilBERT tokenizer to process the text field:
from transformers import (
    DataCollatorWithPadding,
    create_optimizer,
)

# Import metrics
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback
from keras.callbacks import TensorBoard

# Import modules from src
from src.utils.params import train_params

def load_dataset_from_csv(train_path: str, valid_path: str, augument_path: str) -> None:
    """Load the train and valid sets.

    Args:
        train_path (str): Path to the train set.
        valid_path (str): Path to the valid set.
        augument_path (str): Path to the augumented set.

    Returns:
        dataset (dict): Dictionary containing the train and valid sets."""
    if augument_path:
        dataset = load_dataset(
            "csv",
            data_files={"train": [train_path, augument_path], "validation": valid_path},
        )
    else:
        dataset = load_dataset(
            "csv", data_files={"train": train_path, "validation": valid_path}
        )
    return dataset

def preprocess_data(dataset: dict) -> dict:
    """Preprocess the data.

    Args:
        dataset (dict): Dictionary containing the train and valid sets.

    Returns:
        dataset (dict): Dictionary containing cleaned the train and valid sets."""
    dataset = dataset.map(lambda examples: {"text": [text_cleaning(examples["text"])]})
    return dataset
    

def train(params):
    """Pipeline for training the model.

    Args:
        params (dict): Dictionary of parameters.
    
    Returns:
        Saves the trained model to the output directory.
    """

    # Load the train and valid sets
    dataset = load_dataset_from_csv(train_path=params.train_path, 
                                    valid_path=params.valid_path, 
                                    augument_path=params.augument_path)

    # Preprocess the data




if __name__ == "__main__":
    
    # Get start time
    start_time = datetime.now()
    start_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load the parameters from the config file
        with open(sys.argv[1]) as f:
            params_ = train_params(**json.load(f))
    else:
        print("""No config file provided. Specify a config file with the following format:
                { "train_path": "path/to/train.csv",
                  "valid_path": "path/to/valid.csv",
                  "model_name": "distilbert-base-uncased",
                  "output_dir": "path/to/output_dir",
                  "num_labels": 2,
                  "target_label": "label",
                  "augmented_path": "path/to/augmented.csv""")
        sys.exit(1)

    # Create the output directory
    params_.output_dir = f"{params_.output_dir}_{start_time}"
    os.makedirs(params_.output_dir, exist_ok=True)

    # Train the model
    train(params_)
