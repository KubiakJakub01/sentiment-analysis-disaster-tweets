"""
Module for handling parameters
"""

import json
import os
from dataclasses import dataclass, field 
from typing import Optional, List

@dataclass
class TrainParams:
    train_path: str = field(metadata={"help": "Path to the train set."})
    valid_path: str = field(metadata={"help": "Path to the valid set."})
    model_name: str = field(metadata={"help": "Name of the model to use."})
    output_dir: str = field(metadata={"help": "Output directory for the model."})
    num_labels: int = field(metadata={"help": "Number of labels to use."})
    target_label: str = field(metadata={"help": "Target label to use."})
    text_column: str = field(metadata={"help": "Column containing the text to use."})
    remove_columns: Optional[List[str]] = field(default_factory=list, metadata={"help": "Columns to remove from the dataset."})
    augmented_path: Optional[str] = field(default=None, metadata={"help": "Path to the augmented set."})


@dataclass
class Hyperparameters:
    learning_rate: float = field(metadata={"help": "Learning rate to use."})
    metric: str = field(metadata={"help": "Metric to use for the best model."})
    max_length: int = field(metadata={"help": "Maximum length of the input sequence."})
    num_train_epochs: int = field(metadata={"help": "Number of training epochs."})
    batch_size: int = field(metadata={"help": "Batch size for training."})
    weight_decay: float = field(metadata={"help": "Weight decay to use."})
    logging_dir: str = field(metadata={"help": "Directory to save logs."})
    save_strategy: str = field(metadata={"help": "Strategy to use for saving the model."})
    evaluation_strategy: str = field(metadata={"help": "Evaluation strategy to use."})
    eval_steps: int = field(metadata={"help": "Number of steps to evaluate the model."})
    load_best_model_at_end: bool = field(metadata={"help": "Load the best model at the end."})
    greater_is_better: bool = field(metadata={"help": "Whether the metric is greater is better."})


@dataclass
class Params:
    train_params: TrainParams
    hyperparameters: Hyperparameters


def get_params(json_file_path) -> Params:
    """
    Load parameters from json file
    Args:
        json_file_path (str): Path to the json file

    Returns:
        params (Params): Parameters
    """
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            json_loader = json.load(f)
        train_params = TrainParams(**json_loader["train_params"])
        hyperparameters = Hyperparameters(**json_loader["hyperparameters"])
        params = Params(train_params=train_params, hyperparameters=hyperparameters)
    else:
        raise FileNotFoundError(f"File {json_file_path} not found.")
    return params

if __name__ == "__main__":
    params = get_params("src/config/params.json")
    print(params.train_params.train_path)
    print(params.hyperparameters.learning_rate)
