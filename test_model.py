"""
Code for Problem 1 of HW 3.
"""
import pickle

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, AutoTokenizer,\
    Trainer, TrainingArguments

from train_model import preprocess_dataset, compute_metrics, preprocess_dataset_hatexplain

def init_tester(directory: str) -> Trainer:
    """
    Prolem 1f: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """
    test_args = TrainingArguments(
        output_dir = "checkpoints",
        do_train = False,
        do_eval = False,
        do_predict = True,
        dataloader_drop_last = False,
    )
    tester = Trainer(
                model = BertForSequenceClassification.from_pretrained(directory), 
                args = test_args, 
                compute_metrics = compute_metrics)

    return tester

if __name__ == "__main__":  # Use this script to test your model
    model_name = "vinai/bertweet-base"
    labels = 'original'

    # Load dataset
    hatexplain = load_dataset("hatexplain")

    # Preprocess the dataset for the tester
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hatexplain["train"] = preprocess_dataset_hatexplain(hatexplain["train"], tokenizer, labels)

    # Set up tester
    tester = init_tester("models/run-0/checkpoint-500")

    # Test
    results = tester.predict(hatexplain["train"])
    with open("test_results_hatexplain.p", "wb") as f:
        pickle.dump(results, f)