"""
Code for Problem 1 of HW 3.
"""
import pickle

import evaluate
import torch 
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast, BertForSequenceClassification, AutoTokenizer,\
    Trainer, TrainingArguments

from train_model import preprocess_dataset, compute_metrics, preprocess_dataset_hatexplain
from counterfactual_model import CustomTrainer, DebiasBert

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
    model = DebiasBert()
    model.load_state_dict(state_dict = torch.load(directory + '/pytorch_model.bin', map_location=torch.device('cpu')))

    test_args = TrainingArguments(
        output_dir = "checkpoints",
        do_train = False,
        do_eval = False,
        do_predict = True,
        remove_unused_columns=False,
        dataloader_drop_last = False,
    )
    tester = CustomTrainer(
                model = model,
                args = test_args, 
                compute_metrics = compute_metrics)

    return tester

if __name__ == "__main__":  # Use this script to test your model
    model_name = "vinai/bertweet-base"
    labels = 'original'

# Load hate speech and offensive dataset and create validation split
    hate_speech = load_dataset("hate_speech_offensive")
    hatexplain = load_dataset("hatexplain")

    # Preprocess the dataset for the trainer
    labels='original'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    hate_speech["train"] = preprocess_dataset(hate_speech["train"], tokenizer, labels)
    hatexplain["train"] = preprocess_dataset_hatexplain(hatexplain["train"], tokenizer, labels)
    hatexplain["validation"] = preprocess_dataset_hatexplain(hatexplain["validation"], tokenizer, labels)
    hatexplain["test"] = preprocess_dataset_hatexplain(hatexplain["test"], tokenizer, labels)
    
    # concatenate datasets and train val test split
    bert_dataset = concatenate_datasets([hatexplain['train'], hatexplain['validation'], hatexplain['test'], hate_speech['train']])
    remove_cols = [col for col in bert_dataset.column_names if col not in ['demo_props', 'labels', 'input_ids', 'token_type_ids', 'attention_mask',]]
    bert_dataset = bert_dataset.remove_columns(remove_cols)

    split = bert_dataset.train_test_split(.2, seed=3463)

    # Set up tester
    tester = init_tester("checkpoints_debias/checkpoint-1000")

    # Test
    results = tester.predict(split["test"])
    with open("test_results_debias.p", "wb") as f:
        pickle.dump(results, f)