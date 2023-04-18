
"""
Code for Problem 1 of HW 3.
"""
import copy 
import pickle
import torch 
from typing import Any, Dict
import numpy as np
import torch.nn as nn 
import optuna
from datasets import Dataset, load_dataset, load_metric, concatenate_datasets
from transformers import BertModel, Trainer, TrainingArguments
from torch.nn import Linear
from torch.nn.functional import softmax, relu
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertTokenizerFast, BertForSequenceClassification, AutoTokenizer,\
    Trainer, TrainingArguments, EvalPrediction
from train_model import preprocess_dataset, preprocess_dataset_hatexplain


def compute_metrics(eval_preds):
    metric = load_metric('accuracy')
    preds, labels = eval_preds
    preds = np.argmax(preds, axis=1)
    return metric.compute(predictions=preds, references=labels)


class CustomTrainer(Trainer):

    def __init__(self, alpha=0.1, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
        self.beta=beta

    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs.keys())
        print(inputs)
        labels = inputs["labels"]
        logits, factual_logits, counterfactual_logits = model.forward(
            **inputs, training=True
        )
        loss_fct = nn.CrossEntropyLoss()
        counterfactual_loss = loss_fct(counterfactual_logits.view(-1, self.model.num_labels), labels.view(-1))
        factual_loss = loss_fct(factual_logits.view(-1, self.model.num_labels), labels.view(-1))
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        
        # generate a weighted combination of the three losses
        total_loss = self.alpha * counterfactual_loss + self.beta * factual_loss + loss
        return (total_loss, logits) if return_outputs else total_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        modified prediction step otherwise trainer does not 
        """
        with torch.no_grad():
            # move inputs to cuda 
            original_device = inputs["labels"].device.type
            for key in inputs:
                inputs[key] = inputs[key].to(model.bert.device)
            labels = inputs["labels"]
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)

            # put the inputs back to original device 
            for key in inputs:
                inputs[key] = inputs[key].to(original_device)
            if prediction_loss_only:
                return (loss, None, None)
        return loss, logits, labels


class DebiasBert(nn.Module):

    def __init__(self, model_name="vinai/bertweet-base", num_labels=3, hidden_size=128):
        super(DebiasBert, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.num_labels=num_labels

        # copy default embedding layer from BERT 
        self.embedding_layer = copy.deepcopy(self.bert.embeddings)

        # initialize linear layers 
        self.linear = Linear(in_features=128, out_features=1)
        self.linear_context = Linear(in_features=128, out_features=128, bias=True)
        self.linear_token = Linear(in_features=128, out_features=128, bias=True)
        self.classifier = Linear(in_features=128, out_features=3, bias=True)
        

    def forward(self, input_ids, attention_mask=None, training=False, verbose=False, **kwargs):
        
        # use the embedding to generate 
        embeddings = self.embedding_layer(input_ids)
        weights = softmax(torch.squeeze(softmax(self.linear(embeddings), dim=1), dim=2), dim=1)

        # generate a shallow representation of the data 
        avg_token_representation = torch.sum(weights.unsqueeze(-1) * embeddings, axis=1)
        token_input = self.linear_token(avg_token_representation) 
        counterfactual_output = self.classifier(relu(token_input))

        # use BERT model to generate contextual representation
        outputs = self.bert.forward(input_ids, attention_mask=attention_mask)
        context_representation = outputs.last_hidden_state[:,0,:]
        context_input = self.linear_context(context_representation)
        factual_output = self.classifier(relu(context_input + token_input))

        # calculate the difference between factual and counterfactual
        output = factual_output-counterfactual_output
        if training:
            # use all three outputs for training 
            return (
                softmax(output, dim=-1), 
                softmax(factual_output, dim=-1), 
                softmax(counterfactual_output, dim=-1)
            )
        return SequenceClassifierOutput(loss=None, logits=softmax(output, dim=-1))


def init_model(trial: Any, model_name="vinai/bertweet-base", num_labels=3) -> DebiasBert:

    model = DebiasBert(model_name=model_name, num_labels=num_labels)
    return model 


def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset, num_labels=3) -> Trainer:
    training_args = TrainingArguments(
        output_dir="./checkpoints_debias",
        disable_tqdm=False,
        metric_for_best_model='eval_accuracy',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    trainer = CustomTrainer(
        model = None,
        model_init=lambda: init_model(None, model_name, num_labels=num_labels),
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
        'per_device_train_batch_size': [64, 128],
        'learning_rate': [3e-4, 1e-4, 5e-5, 3e-5],
        'num_train_epochs': [8],
        'seed': [3463]
    }

    def my_hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [3e-4, 1e-4, 5e-5, 3e-5]),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [64, 128]),
        }

    return {
        'direction': "maximize",
        'backend': "optuna",
        'n_trials': 8,
        'compute_objective': lambda metrics: metrics['eval_accuracy'],
        'hp_space': my_hp_space,
        'sampler': optuna.samplers.GridSampler(search_space),
    }

if __name__ == "__main__":  # Use this script to train your model
    model_name = "vinai/bertweet-base"
    
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
    
    split = bert_dataset.train_test_split(.2, seed=3463)
    split_2 = split["train"].train_test_split(.125, seed=3463)
    split["train"] = split_2["train"]
    split["val"] = split_2["test"]

    # Set up trainer
    trainer = init_trainer(model_name, split["train"], split["val"])

    # Train and save the best hyperparameters   
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    with open("train_results_debias_bert.p", "wb") as f:
        pickle.dump(best, f)