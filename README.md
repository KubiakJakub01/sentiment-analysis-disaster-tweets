# Sentiment Analysis of Disaster Tweets
## Intro

This repository contains code for Kaggle competition: [NLP-with-Disaster-Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview). In this competition we have to predict which Tweets are about real disasters and which ones are not. The dataset might be found in data section [NLP-with-Disaster-Tweets/Data](https://www.kaggle.com/competitions/nlp-getting-started/data).

## Setup

In this repository I based on Tensorflow 2.X and HuggingFace transformers. Start setting up repository with:

You can clone this repository with submodules with fine-tuned models using this command: 

```git clone --recurse-submodules git@github.com:KubiakJakub01/Kaggle-NLP_with-Disaster-Tweets.git```

or you can clone this repository and then download submodules with:

```git clone https://github.com/KubiakJakub01/sentiment-analysis-disaster-tweets.git``` 

Now you can install all requirements in virtual environment. I recommend to use [conda](https://docs.conda.io/en/latest/miniconda.html) because it make easier to install all necessary packages and dependencies especially for GPU support. So first install miniconda and then create virtual environment with:

```conda create -n venv python=3.9```

```conda activate venv```

And them you can install all requirements with:

```pip install -r requirements.txt```

### HuggingFace models repository

I used my [HuggingFace](https://huggingface.co/KubiakJakub01) repo to download pretrained models. You can create sub-module with models with:

```git clone https://huggingface.co/KubiakJakub01/finetuned-distilbert-base-uncased```

or you can use this models with transformers libraries from HuggingFace. Here is code sniped how to do it in code:

```
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("KubiakJakub01/finetuned-distilbert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained("KubiakJakub01/finetuned-distilbert-base-uncased")
```

For more information about HuggingFace models you can check [HuggingFace documentation](https://huggingface.co/transformers/model_doc/auto.html).

### GPU support

If you want to use GPU support you need to install CUDA and cuDNN. I recommend to use [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN](https://developer.nvidia.com/cudnn) for CUDA 11.0. You can find more information about installation [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).
To check if you have installed CUDA and cuDNN correctly you can run:

```nvcc --version``` <> to check CUDA version

```nvidia-smi``` <> to check cuDNN version

```python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"``` <> to check if GPU is available in Tensorflow

### Docker

You can also use Docker to run this repository. First you need to install [Docker](https://docs.docker.com/get-docker/). Then you can build Docker image with:

```docker build -t sentiment-analysis-disaster-tweets .```


## Usage

### Explore data

Before you start building, it's a good idea to look at the data you have. To do this you can use ```src/notebooks/research.ipynb``` which shows the basic statistics of the data and their distribution. Notebook not only focuses on text data but also analyzes other features.

### Split data (optional)

Before training you can manually divide the data into traning and validation and leave only the necessary columns, i.e. text and target. You can do it with this script:

```
python -m src.data_preprocessing.split_data --data_path path/to/data \
                                                 [--train_size 0.9] \
                                                [--target_column target] \
                                                [--text_column text] \
                                                [--random_state 42]
```

As a result, you will get train and valid files in csv form.
### Fine-tune model

The main script responsible for traning the model is the ```src/train.py```. Before running the script, you need to prepare the configuration file. An example file can be found ```src/config/params.json```. A description of all parameters, their types and default values can be found ```src/utils/params.py```.
With the configuration file ready, you can run the script with the command:

```
python3 -m src.train src/config/params.json
```

As a result, you will get direcotry with trained named as ```model_save_name```. In addition, a folder will be created with tensorboard training logs in ```output_dir```.

### Inference

To check performance of the model you can use ```src/eval.py``` script. You can run it with:

```
python src/eval.py -m models/bert-base-uncased \ 
                    [-t data/test.csv] \
                    [-s results] \
                    [-b 8] \
                    [-n 2] \ 
                    [-e accuracy precision recall f1] \
                    [--target_column target] \
                    [--text_column text] \
                    [--id_column id]
```

As a result, you will get csv file with predictions and metrics. Which will be save in ```--save_predictions_path```.

### Predict

To predict on new data you can use ```src/predict.py``` script. You can run it with:

```
python src/get_predictions.py -m models/bert-base-uncased \ 
                        -s results \
                        [-t data/test.csv] \
                        [-n 2] \
                        [-b 8] \
                        [--text_column text] \
                        [--id_column id]
```

As a result, you will get csv file with predictions. Which will be save in ```--save_predictions_path```.

### Inference notebook

You can also use ```src/notebooks/inference.ipynb``` to check performance of the model. You can also use it to predict on new data.

### Data augumentation

To augument data you can use ```src/data_preprocessing/augument_data.py``` script. You can run it with:

```
python src/data_preprocessing/data_augumentation.py [-p data/train.csv] \
                                                [-s data/augumented_train.csv] \
                                                [-m models/bert-base-uncased] \
                                                [-a substitute] \
                                                [-n 2] \
                                                [--aug_min 1] \
                                                [--aug_p 0.3]
```

As a result, you will get csv file with augumented data. Which will be save in ```--path_to_save_data```.

## Results

I made experiments for two DistilBERT base uncased model with different dataset and slightly altered hyperparameters.

### models/finetuned-distilbert-base-uncased

In fist experiment I based one only on dataset from kaggle. I fine-tuned DistilBERT base uncased model with default hyperparameters. I used AdamW optimizer with learning rate 2e-5 and 3 epochs. I used 10% of data for validation. I used accuracy, precision, recall and f1 metrics. I got the following results:

| Metric | Value |
| --- | --- |
| Accuracy | 0.827 |
| Precision | 0.815 |
| Recall | 0.786 |
| F1 | 0.800 |

For more information about model you can check huggingface [model card](https://huggingface.co/KubiakJakub01/finetuned-distilbert-base-uncased).
### models/finetuned-distilbert-base-augumented

In second experiment beyond kaggle dataset I used augumented. I fine-tuned DistilBERT base uncased model. I used this same hyperparameters as above. I used 10% of data for validation. I used accuracy, precision, recall and f1 metrics. I got the following results:  

| Metric | Value |
| --- | --- |
| Accuracy | 0.798 |
| Precision | 0.801 |
| Recall | 0.720 |
| F1 | 0.759 |

For more information about model you can check huggingface [model card](https://huggingface.co/KubiakJakub01/finetuned-distilbert-base-augumented).

### Submission

I made submission with model both models and got the following results: 

    * models/finetuned-distilbert-base-uncased: 0.81857
    * models/finetuned-distilbert-base-augumented: 0.80849

The results are very close to each other. It is interesting that augmentation of the data did not improve the results because in this variant there was twice as much training data. For the moment, the results of both models are in the top 20% of the ladder

## Future work

I started experimenting with adding layers to the model ```src/model/utils/build_custom_transformer.py``` and over tuning of training hyperparameters. However, I have not yet tested these ideas. I also plan to try other models, such as BERT, RoBERTa, XLNet, etc. I also plan to try to use other data augmentation techniques, such as backtranslation, etc. 

