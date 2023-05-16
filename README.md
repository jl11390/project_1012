# Using Counterfactuals to Debias Hate Speech Detection

Toxic comment classification models including BERT have shown promising results in detecting harmful and offensive comments on online communication platforms. However, some models lack consideration of demographic language habits and thus may exhibit biases against African American English (AAE) speakers. However, the data has found to exhibit biases against African American English (AAE) language, resulting in trained models that propagate the same bias. Although an abundance of research attempts to uncover and mitigate biases in machine learning systems, addressing these biases typically involves increasing the dataset size and significantly increasing the computational complexity, and may only effectively debias a limited number of predetermined groups. We propose a Debiased BERT model that draws on counterfactual inference techniques to devise methods for mitigating bias in hate social media posts classification tasks. Our model outperforms the basic BERT model on 5 different AAE discrimination metrics.

<!-- <p align="center">
  <img width="700" height="624" src="">
</p> -->

| Method         | F1    | F1_AAE | DI_toxic | SPD    | EOD    | AOD    | PP     |
| -------------- | ----- | ------ | -------- | ------ | ------ | ------ | ------ |
| Baseline BERT  | 0.750 | 0.873  | 1.313    | -0.231 | -0.144 | -0.264 | -0.091 |
| Debiased BERT  | **0.805** | **0.885**  | **1.199**    | **-0.164** | **-0.051** | **-0.223** | **-0.077** |


## Prerequisites

* Python (3.9.16)
* PyTorch (1.8.0)
* torchvision (0.9.0)
* NumPy (1.23.5)
* transformers (4.24.0)
* pandas (2.0.0)
* tqdm (4.65.0)
* nltk (3.7)

## How to run the code

```python
python counterfactual_model.py
```


