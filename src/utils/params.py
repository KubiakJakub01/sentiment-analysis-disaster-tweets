"""
Module for handling parameters
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainParams:
    train_path: str = field(metadata={"help": "Path to the train set."})
    valid_path: str = field(metadata={"help": "Path to the valid set."})
    model_name: str = field(metadata={"help": "Name of the model to use."})
    output_dir: str = field(metadata={"help": "Output directory for the model."})
    num_labels: int = field(metadata={"help": "Number of labels to use."})
    target_label: str = field(metadata={"help": "Target label to use."})
    augmented_path: Optional[str] = field(default=None, metadata={"help": "Path to the augmented set."})


@dataclass
class Hyperparameters:
    learning_rate: float
    max_length: int
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    logging_dir: str
    logging_steps: int
    save_steps: int
    save_total_limit: int
    evaluation_strategy: str
    eval_steps: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool


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
            params = Params(**json.load(f))
    else:
        raise FileNotFoundError(f"File {json_file_path} not found.")
    return params
