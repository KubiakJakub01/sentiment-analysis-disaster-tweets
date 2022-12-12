"""
Split the data into train and valid sets.
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split the data into train and valid sets."
    )
    parser.add_argument(
        "--data_path", type=str, default="data/data.csv", help="Path to the data file."
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.9,
        help="Proportion of the data to include in the train split.",
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state for reproducibility."
    )
    args = parser.parse_args()
    return args


def split_data(data_path: str, train_size: int = 0.8, random_state: int = 42) -> None:
    """Split the data into train and valid sets.

    Args:
        data_path (str): Path to the data file.
        train_size (float): Proportion of the data to include in the train split.
        random_state (int): Random state for reproducibility.

    Returns:
        None: Saves the train and valid sets to disk.
    """
    # Read in the data
    data_df = pd.read_csv(data_path)

    # Split the data into train and valid sets
    train, valid = train_test_split(
        data_df, train_size=train_size, random_state=random_state
    )

    # Data base dir
    data_base_dir = Path(data_path).parent
    # Save the train and test sets
    train.to_csv(f"{data_base_dir}/train.csv", index=False)
    valid.to_csv(f"{data_base_dir}/valid.csv", index=False)

    print(f"DONE: Train and valid sets saved to {data_base_dir}.")


if __name__ == "__main__":
    params = parse_args()
    split_data(
        data_path=params.data_path,
        train_size=params.train_size,
        random_state=params.random_state,
    )
