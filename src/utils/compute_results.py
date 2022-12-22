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
    