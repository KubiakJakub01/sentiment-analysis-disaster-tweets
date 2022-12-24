"""
Augument data by adding noise to the data
"""

# Import basic libraries
import argparse

import nlpaug.augmenter.word as nlpaw
import pandas as pd
import tqdm


def get_params():
    """Get parameters from command line.
    
    Returns:
        args (argparse.Namespace): Arguments from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_data",
        "-p",
        type=str,
        default="data/train.csv",
        help="Path to the data to augument.",
    )
    parser.add_argument(
        "--path_to_save_data",
        "-s",
        type=str,
        default="data/augumented_train.csv",
        help="Path to save the augumented data.",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="bert-base-uncased",
        help="Model name.",
    )
    parser.add_argument(
        "--augumentation_type",
        "-a",
        type=str,
        default="substitute",
        help="Augumentation type to use.",
    )
    return parser.parse_args()
