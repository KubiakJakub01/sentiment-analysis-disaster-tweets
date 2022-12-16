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
from src.utils.params import get_params
# Import utils for text cleaning
from src.utils.text_cleaning import text_cleaning
# Import model and tokenizer selector class
from src.model.nlp_models_selector import get_model_and_tokenizer


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


def tokenize_text(text):
    """Tokenize the data.

    Args:
        text (str): Text to tokenize.

    Returns:
        tokenized_data (dict): Dictionary containing the tokenized data."""
    return tokenizer(text["text"], truncation=True, is_split_into_words=True)


def prepare_dataset(dataset, columns, label_cols, batch_size, shuffle, collate_fn):
    """Prepare the dataset for training.

    Args:
        dataset (dict): Dictionary containing the train and valid sets.
        columns (list): List of columns to use.
        label_cols (list): List of labels to use.
        batch_size (int): Batch size to use.
        shuffle (bool): Whether to shuffle the data.
        collate_fn (function): Function to use for collating the data.

    Returns:
        dataset (dict): Dictionary containing the train and valid sets."""
    tf_dataset = dataset.to_tf_dataset(
        columns=columns,
        label_cols=label_cols,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return tf_dataset


def train():
    """Pipeline for training the model.

    Args:
        params (dict): Dictionary of parameters.
    
    Returns:
        Saves the trained model to the output directory.
    """

    # Load the train and valid sets
    dataset = load_dataset_from_csv(train_path=params.train_params.train_path, 
                                    valid_path=params.train_params.valid_path, 
                                    augument_path=params.train_params.augument_path)

    # Preprocess the data
    dataset = preprocess_data(dataset)

    # Tokenize the data
    tokenized_dataset = dataset.map(
        tokenize_text,
        batched=True,
        remove_columns=["id", "keyword", "location", "text"],
    )

    # Load data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    # Prepare the dataset for training
    tf_train_dataset = prepare_dataset(
        tokenized_dataset["train"],
        columns=["input_ids", "attention_mask", params.train_params.target_label],
        label_cols=[params.train_params.target_label],
        batch_size=params.hyperparameters.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

if __name__ == "__main__":
    
    # Get start time
    start_time = datetime.now()
    start_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Load parameters from config json file
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load the parameters from the config file
        params = get_params(sys.argv[1])
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

    # Load model, and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_name=params.train_params.model_name, 
                                               num_labels=params.train_params.num_labels)

    # Create the output directory
    params.train_params.output_dir = f"{params.train_params.output_dir}_{start_time}"
    os.makedirs(params.train_params.output_dir, exist_ok=True)

    # Train the model
    train(params)
