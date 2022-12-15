"""
Module for handling parameters
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class train_params:
    train_path: str = field(metadata={"help": "Path to the train set."})
    valid_path: str = field(metadata={"help": "Path to the valid set."})
    model_name: str = field(metadata={"help": "Name of the model to use."})
    output_dir: str = field(metadata={"help": "Output directory for the model."})
    num_labels: int = field(metadata={"help": "Number of labels to use."})
    target_label: str = field(metadata={"help": "Target label to use."})
    augmented_path: Optional[str] = field(default=None, metadata={"help": "Path to the augmented set."})
