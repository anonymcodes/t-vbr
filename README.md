# Learning Temporal Representation for Grocery Recommendation

>This is our Pytorch implementation for the paper:
>**Anonymous Author(s), Temporal Variational Bayesian Representation Learning for Grocery Recommendation, 2020**

## Introduction

This paper focuses on the recommendation task in the grocery shopping scenario, where users purchase multiple products in sequential baskets. It is commonly acknowledged that both users' interests and products' popularities vary over time. However, few prior grocery recommendation methods account for such temporal patterns, instead, representing each user and product via a static low dimensional vector. In contrast, we propose a new model: Temporal Variational Bayesian Representation (T-VBR) for grocery recommendation, which is able to encode temporal patterns to improve effectiveness. T-VBR is designed under the temporal variational Bayesian framework, and it learns the temporal Gaussian representations for users and items by encoding information from: 1) the basket context; 2) item side information; and 3) the temporal context from past user-item interactions. T-VBR is trained using sampled triples of users with two items bought together in baskets during different time windows, via a Bayesian Skip-gram model based on a temporal variational auto-encoder. Experiments conducted on four public grocery shopping datasets show that our proposed T-VBR model can significantly outperform the existing state-of-the-art grocery recommendation methods, and can learn more expressive representations that effectively capture the temporal information.

## Environment Requirement

The code has been tested running under Python 3.7.5 and Pytorch 1.4.0. The required packages can be found at requirements.txt

## Usage instruction


To run our models as well as the baselines, you can go to the "./example" floder, run the python code on terminal by:

```shell
    python ./train_tvbr.py --dataset dunnhumby --n_sample 1000000 --emb_dim 64
```
## All the default paramter settings are in "./configs" 
## You can also specify you customized arguments by command line:

--dataset: Specify the datasets.

--percent: Percentage of the dataset. Only used in Instacart dataset.

--n_sample: Number of sampled triples.

--emb_dim: Dimension of embeddings 

--lr: Inital learning rate

--alpha: Parameter for the KL terms. We use the annealing technique to decay the impact of KL terms.

--time_step: Time step for spliting the sequential baskets

--max_epoch: Training epoches. 

--item_fea_type: Item feature type. Can be 'random' 'word2vec' 'bert' or their combinations.
