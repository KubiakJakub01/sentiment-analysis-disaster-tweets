"""
Main inference script for the project.
"""

# Imports basic libraries
import argparse
import json
from datetime import datetime
from pathlib import Path

# Import huggingface libraries
from datasets import load_dataset
from transformers import pipeline

# Import modules
from src.model.nlp_models_selector import get_model_and_tokenizer
from src.utils.compute_results import get_results, save_results
from src.utils.nlp_metric import Metric


def get_params():
    """Get parameters from command line.
    
    Returns:
        args (argparse.Namespace): Arguments from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        help="Path to checkpoint with the model to use.",
    )
    parser.add_argument(
        "--path_to_test_data",
        "-t",
        type=str,
        default="data/test.csv",
        help="Path to the test set.",
    )
    parser.add_argument(
        "--num_labels",
        "-n",
        type=int,
        default=2,
        help="Number of labels for the model.",
    )
    parser.add_argument(
        "--save_predictions_path", "-s", type=str, help="Path to save the predictions."
    )
    parser.add_argument(
        "--metrics",
        "-e",
        nargs="+",
        type=str,
        help="Metrics to use.",
        default=["accuracy", "precision", "recall", "f1"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    
    # Get parameters
    params = get_params()

    # Define constant variables
    MODEL_PATH = Path(params.model_path)
    MODEL_NAME = MODEL_PATH.name
    NUM_LABELS = params.num_labels
    START_TIME = datetime.now()
    START_TIME = START_TIME.strftime("%Y-%m-%d_%H-%M-%S")
    SAVE_PREDICTIONS_PATH = Path(params.save_predictions_path) / MODEL_NAME / START_TIME
