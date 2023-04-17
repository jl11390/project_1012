
"""
Code for Problem 1 of HW 3.
"""
import copy 
import torch 
from typing import Any, Dict
import numpy as np
import torch.nn as nn 
from datasets import Dataset, load_dataset, load_metric
from transformers import BertModel, Trainer, TrainingArguments
from torch.nn import Linear
from torch.nn.functional import softmax, relu


class CustomTrainer(Trainer):

    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha

    def compute_loss(self, model, inputs):
        labels = inputs["labels"]
        counterfactual_logits = model.forward(**inputs, counter_factual=True)
        factual_logits = model.forward(**inputs, counter_factual=False)
        loss_fct = nn.CrossEntropyLoss()
        
        counterfactual_loss = loss_fct(counterfactual_logits.view(-1, self.model.num_labels), labels.view(-1))
        factual_loss = loss_fct(factual_logits.view(-1, self.model.num_labels), labels.view(-1))
        return counterfactual_loss + self.alpha * factual_loss


def compute_metrics(eval_preds):
    metric = load_metric('accuracy')
    preds, labels = eval_preds
    preds = np.argmax(preds, axis=1)
    return metric.compute(predictions=preds, references=labels)


class DebiasBert(nn.Module):

    def __init__(self, model_name="vinai/bertweet-base", num_labels=3, hidden_size=128, **kwargs):
        super(DebiasBert, self).__init__(**kwargs)
        self.bert = BertModel.from_pretrained(model_name)
        self.num_labels=num_labels

        # copy default embedding layer 
        self.embedding_layer = copy.deepcopy(self.bert.embeddings)

        # initialize linear layers 
        self.linear = Linear(in_features=128, out_features=1)
        self.linear_context = Linear(in_features=128, out_features=hidden_size, bias=True)
        self.linear_token = Linear(in_features=128, out_features=hidden_size, bias=True)
        self.classifier = Linear(in_features=hidden_size, out_features=num_labels, bias=True)
        

    def forward(self, input_ids, attention_mask=None, counter_factual=True, verbose=False, **kwargs):

        # use BERT model to generate contextual representation
        outputs = self.bert.forward(input_ids, attention_mask=attention_mask)
        context_representation = outputs.last_hidden_state[:,0,:]
        context_input = self.linear_context(context_representation)
        
        # only use contextual representation
        if counter_factual:
            return softmax(self.classifier(relu(context_input)), dim=-1)
        
        # convert each input into an embedding 
        embeddings = self.embedding_layer(input_ids)
        weights = torch.nn.functional.softmax(
            torch.squeeze(torch.nn.functional.softmax(
                self.linear(embeddings), dim=1), dim=2), dim=1)
        
        # generate shallow features from input text 
        avg_token_representation = torch.sum(weights.unsqueeze(-1) * embeddings, axis=1)
        token_input = self.linear_token(avg_token_representation) 
        classifier_output = self.classifier(relu(context_input + token_input))
        if verbose:
            print(classifier_output.shape)
        return softmax(classifier_output, dim=-1)


def init_model(trial: Any, model_name="vinai/bertweet-base", num_labels=3) -> DebiasBert:

    model = DebiasBert(model_name=model_name, num_labels=num_labels)
    return model 


def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset, num_labels=3) -> Trainer:
    training_args = TrainingArguments(
        output_dir="./checkpoints",
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