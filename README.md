# Using Counterfactuals to Debias Hate Speech Detection

Toxic comment classification models including BERT have shown promising results in detecting harmful and offensive comments on online communication platforms. However, some models lack consideration of demographic language habits and thus may exhibit biases against African American English (AAE) speakers. However, the data has found to exhibit biases against African American English (AAE) language, resulting in trained models that propagate the same bias. Although an abundance of research attempts to uncover and mitigate biases in machine learning systems, addressing these biases typically involves increasing the dataset size and significantly increasing the computational complexity, and may only effectively debias a limited number of predetermined groups. We propose a Debiased BERT model that draws on counterfactual inference techniques to devise methods for mitigating bias in hate social media posts classification tasks. Our model outperforms the basic BERT model on 5 different AAE discrimination metrics.

To address this issue, it is crucial that we actively work to reduce biases within the model. In this work, we aim to evaluate whether causal approaches can reduce bias for the toxicity detection task by creating a model that not only learns from observation but also learns from the process of counterfactual predictions. We propose a new model architecture that concurrently learns two representations of the input text. We have one module that derives shallow features from the text, such as those keywords within the text, whilst the other develops a contextual representation through BERT. By contrasting the difference in its predictions with access to either one or both representations, we can gauge the ratio of the overall output derived from surface-level characteristics, versus a prediction that depends on the model's understanding of the text. We believe that such an approach can enable more automated and less racially biased text classification systems. Additionally, we want to verify whether this approach can be utilized without significantly affecting the overall performance. 

<p align="center">
  <img width="700" height="624" src="https://github.com/jl11390/project_1012/blob/main/figures/nework-diagram.png">
</p>

The Debiased BERT model consistently outperforms the basic BERT model in fairness metrics, indicating its greater effectiveness in reducing biases and ensuring equitable treatment across various groups. Notably, $DI_{\text{unfav}}$ lies within the acceptable range of 0.8-1.2 for the Debiased BERT model, which is not the case for BERT. Additionally, SPD, EOD, and AOD results of Debiased BERT are all closer to 0 compared to BERT, signifying a more balanced distribution of outcomes and opportunities between AAE group and non-AAE group. Moreover, the narrower disparity in PP highlights that the Debiased BERT model achieves better calibration between predicted probabilities and actual outcomes. 


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


