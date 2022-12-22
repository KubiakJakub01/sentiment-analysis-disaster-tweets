"""
Compute results based on the predictions and labels.
"""
# Imports basic libraries
import argparse
import json
from pathlib import Path

import pandas as pd

# Import modules
from src.utils.nlp_metric import Metric


def get_params():
    """Get the parameters from command line.
    
    Returns:
        args (argparse.Namespace): Arguments from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_labels", "-l", type=str, help="Path to the labels.")
    parser.add_argument(
        "--path_to_predictions", "-p", type=str, help="Path to the predictions."
    )
    parser.add_argument(
        "--path_to_save_results", "-s", type=str, help="Path to save the results."
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        type=str,
        help="Metrics to use.",
        default=["accuracy", "precision", "recall", "f1"],
    )
    return parser.parse_args()


def read_predictions(path_to_predictions):
    """Read predictions from json file.
    
    Args:
        path_to_predictions (str): Path to the predictions.
    
    Returns:
        preds (dict): Dictionary containing the predictions."""
    with open(path_to_predictions, "r") as f:
        preds = json.load(f)
    return preds


def read_labels(path_to_labels):
    """Read labels from a csv file.
    
    Args:
        path_to_labels (str): Path to the labels.
        
    Returns:
        labels (list): List containing the labels."""
    df = pd.read_csv(path_to_labels)
    labels = df["labels"].tolist()
    return labels
