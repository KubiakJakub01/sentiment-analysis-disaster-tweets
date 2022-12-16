"""
Metrics for NLP tasks.
"""
# Path: src\utils\NLP_metric.py
# import libraries
from dataclasses import dataclass
import numpy as np

# import metrics
import evaluate


@dataclass
class Metric:
    """Class to compute the metric for the model."""

    def __init__(self, metric_name: str):
        """Initialize the metric."""
        self.metric_name = metric_name
        self.metric = evaluate.load(metric_name)

    def format(func):
        """Format decorator."""

        def wrapper(*args, **kwargs):
            """Format the results."""
            computed_results = func(*args, **kwargs)
            computed_value = list(computed_results.values())[0]
            return "{:.3f}".format(computed_value)

        return wrapper

    def compute_metrics(self, eval_predictions):
        """Compute the metrics."""
        predictions, labels = eval_predictions
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)

    @format
    def compute(self, predictions, references):
        """Compute the metric."""
        return self.metric.compute(predictions=predictions, references=references)
