"""
Script to find optimal hyperparameters for model fine-tuning
"""
# Import tensorflow, wandb and transformers libraries
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

# Import modules from src
from src.model.nlp_models_selector import model_fn


# Define the input function for training and evaluation
def input_fn(x, y, batch_size):
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_len)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    return dataset

if __name__ == "__main__":
    # Initialize Wandb
    wandb.init(project="hyperparameter-optimization")

    # Define the search space for the hyperparameters
    config = {
        'model_type': wandb.config.Choice('distilbert-base-uncased', ['distilbert-base-uncased', 'bert-base-uncased']),
        'learning_rate': wandb.config.LogUniform(1e-5, 1e-2)
    }

    # Create the model
    model = tf.keras.estimator.model_to_estimator(model_fn, config=config)

    # Train the model with Wandb
    model.train(input_fn=lambda: 
                input_fn(x_train, y_train, batch_size), 
                steps=steps, callbacks=[WandbCallback()])

    # Evaluate the model with Wandb
    model.evaluate(input_fn=lambda: input_fn(x_test, y_test, batch_size))

    # Use Wandb to search for the optimal set of hyperparameters
    best_config = wandb.hyperparameter.search(model, config, objective='val_accuracy')
    print(best_config)
