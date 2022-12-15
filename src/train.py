"""
Main training script for the project.
"""

# Import basic libraries
import argparse
from datetime import datetime
from datasets import load_dataset

# Load the DistilBERT tokenizer to process the text field:
from transformers import (
    DataCollatorWithPadding,
    create_optimizer,
)

# Import metrics
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback
from keras.callbacks import TensorBoard

def get_params():
    """Get the parameters for the training.

    Returns:
        args (argparse.Namespace): Namespace containing the parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        "-t",
        type=str,
        default="data/train.csv",
        help="Path to the train set.",
    )
    parser.add_argument(
        "--valid_path",
        "-v",
        type=str,
        default="data/valid.csv",
        help="Path to the valid set.",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="distilbert-base-uncased",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="output",
        help="Output directory for the model.",
    )
    parser.add_argument(
        "--augumented_path",
        "-a",
        type=str,
        default="data/augumented.csv",
        help="Path to the augumented set.",
        required=False,
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of labels to use.",
        required=False,
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default="label",
        help="Target label to use.",
        required=False,
    )
    args = parser.parse_args()
    return args


