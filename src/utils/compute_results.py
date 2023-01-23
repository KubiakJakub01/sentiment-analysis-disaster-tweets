"""
Compute results based on the predictions and labels.
"""
# Imports basic libraries
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

# Import modules
from src.utils.nlp_metric import Metric

# Set logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
        "--text_column",
        "-t",
        type=str,
        default="text",
        help="Name of the column with the text.",
    )
    parser.add_argument(
        "--id_column",
        "-i",
        type=str,
        default="id",
        help="Name of the column with the id.",
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


def read_predictions(path_to_predictions: str) -> dict:
    """Read predictions from json file.

    Args:
        path_to_predictions (str): Path to the predictions.

    Returns:
        preds (dict): Dictionary containing the predictions."""
    with open(path_to_predictions, "r", encoding="utf-8") as f:
        preds = json.load(f)
    return preds


def read_labels(path_to_labels: str, target_column: str) -> list:
    """Read labels from a csv file.

    Args:
        path_to_labels (str): Path to the labels.
        target_column (str): Name of the column with the labels.

    Returns:
        labels (list): List containing the labels."""
    df = pd.read_csv(path_to_labels)
    labels = df[target_column].tolist()
    return labels


def get_results(preds: list, labels: list, metrics: list) -> dict:
    """Get results for the model.

    Args:
        preds (list): List of predictions.
        labels (list): List of labels.
        metrics (list): List of metrics to use.

    Returns:
        results (dict): Dictionary with the results."""
    results = {}
    preds_labels = [pred["target"] for pred in preds]
    for metric in metrics:
        results[metric.metric_name] = metric.compute(preds_labels, labels)
    return results


def save_results(results: dict, save_predictions_path: str) -> None:
    """Save the results in a json file.

    Args:
        results (dict): Dictionary with the results.
        save_predictions_path (str): Path to save the results."""
    if save_predictions_path:
        with open(save_predictions_path / "metrics.txt", "w", encoding="utf-8") as f:
            f.write(str(results))


if __name__ == "__main__":
    # Get parameters
    params = get_params()

    logger.info("Path to predictions: %s", params.path_to_predictions)
    # Load predictions
    preds = read_predictions(Path(params.path_to_predictions))

    logger.info("Path to labels: %s", params.path_to_labels)
    # Load labels
    labels = read_labels(Path(params.path_to_labels))

    # Load metrics
    metrics = [Metric(metric) for metric in params.metrics]

    logger.info("Compute results...")
    # Compute metrics
    results = get_results(preds, labels, metrics)

    # Print metric results
    logger.info("Results: %s", results)

    logger.info("Path to save results: %s", params.path_to_save_results)
    # Save metric results
    save_results(results, Path(params.path_to_save_results))
