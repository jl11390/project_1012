"""
Code for Problem 1 of HW 3.
"""
import pickle
from typing import Any, Dict

import evaluate
import numpy as np
import optuna
from datasets import Dataset, load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, AutoTokenizer,\
    Trainer, TrainingArguments, EvalPrediction
from functools import partial
from label_aae import predict, load_model

def convert_class(c: int):
    if c == 2:
        return {'labels': 0}
    else:
        return {'labels': 1}

def _demo(tweet: str):
    return {'demo_props': predict(tweet.split())}

def preprocess_dataset(dataset: Dataset, tokenizer: AutoTokenizer) \
        -> Dataset:
    """
    Problem 1d: Implement this function.

    Preprocesses a dataset using a Hugging Face Tokenizer and prepares
    it for use in a Hugging Face Trainer.

    :param dataset: A dataset
    :param tokenizer: A tokenizer
    :return: The dataset, prepreprocessed using the tokenizer
    """ 
    load_model()
    dataset = dataset.map(lambda d: _demo(d['tweet']))
    dataset = dataset.map(lambda d: convert_class(d['class']))
    return dataset.map(lambda d: tokenizer(d['tweet'], padding="max_length", truncation=True))


def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> \
        BertForSequenceClassification:
    """
    Problem 1e: Implement this function.

    This function should be passed to your Trainer's model_init keyword
    argument. It will be used by the Trainer to initialize a new model
    for each hyperparameter tuning trial. Your implementation of this
    function should support training with BitFit by freezing all non-
    bias parameters of the initialized model.

    :param trial: This parameter is required by the Trainer, but it will
        not be used for this problem. Please ignore it
    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be loaded
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A newly initialized pre-trained Transformer classifier
    """
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if use_bitfit:
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.requires_grad = False
            elif 'bias' in name:
                param.requires_grad = True
            else:
                raise Exception
    
    return model

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset,
                 use_bitfit: bool = False) -> Trainer:
    """
    Prolem 1f: Implement this function.

    Creates a Trainer object that will be used to fine-tune a BERT-tiny
    model on the IMDb dataset. The Trainer should fulfill the criteria
    listed in the problem set.

    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be fine-tuned
    :param train_data: The training data used to fine-tune the model
    :param val_data: The validation data used for hyperparameter tuning
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A Trainer used for training
    """
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        # num_train_epochs=4,
        # evaluation_strategy="steps"
        )
    trainer = Trainer(
        model = None,
        model_init=lambda: init_model(None, model_name, use_bitfit),
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )
    return trainer


def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Problem 1g: Implement this function.

    Returns keyword arguments passed to Trainer.hyperparameter_search.
    Your hyperparameter search must satisfy the criteria listed in the
    problem set.

    :return: Keyword arguments for Trainer.hyperparameter_search
    """
    search_space = {
        'per_device_train_batch_size': [64],
        'learning_rate': [3e-4, 1e-4, 5e-5, 3e-5],
        'num_train_epochs': [4],
        'seed': [3463]
    }

    def my_hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [3e-4, 1e-4, 5e-5, 3e-5]),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [64]),
        }

    return {
        'direction': "maximize",
        'backend': "optuna",
        'n_trials': 20,
        'compute_objective': lambda metrics: metrics['eval_accuracy'],
        'hp_space': my_hp_space,
        'sampler': optuna.samplers.GridSampler(search_space),
    }

if __name__ == "__main__":  # Use this script to train your model
    model_name = "vinai/bertweet-base"

    # Load hate speech and offensive dataset and create validation split
    hate_speech = load_dataset("hate_speech_offensive")
    split = hate_speech["train"].train_test_split(.3, seed=3463)
    hate_speech["train"] = split["train"]

    split = hate_speech["train"].train_test_split(.333, seed=3463)
    hate_speech["train"] = split["train"]
    hate_speech["val"] = split["test"]

    # Preprocess the dataset for the trainer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    hate_speech["train"] = preprocess_dataset(hate_speech["train"], tokenizer)
    hate_speech["val"] = preprocess_dataset(hate_speech["val"], tokenizer)

    # Set up trainer
    trainer = init_trainer(model_name, hate_speech["train"], hate_speech["val"])

    # Train and save the best hyperparameters   
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    with open("train_results.p", "wb") as f:
        pickle.dump(best, f)
