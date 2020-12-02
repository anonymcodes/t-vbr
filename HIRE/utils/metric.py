#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tabulate
from sklearn.metrics import mean_squared_error

DEFAULT_USER_COL = "col_user"
DEFAULT_ITEM_COL = "col_item"
DEFAULT_RATING_COL = "col_rating"
DEFAULT_LABEL_COL = "col_label"
DEFAULT_ORDER_COL = "col_order"
DEFAULT_FLAG_COL = "col_flag"
DEFAULT_TIMESTAMP_COL = "col_timestamp"
DEFAULT_PREDICTION_COL = "col_prediction"
from .evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k, rmse
import sys



def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)
    # return mean_squared_error(pred, actual)


# you can add other metric here


def evaluate(data_df, U, V):
    """Evaluate the performance of a prediction by different metrics.
    Args:
        data_df (DataFrame): the dataset to be evaluated.
        predictions (narray): 1-D array. The predicted scores for each user-item pair in the dataset.
        metrics (list):  metrics to be evaluated.
        k_li (int or list): top k (s) to be evaluated.
    Returns:
        result_dic (dict): Performance result.
    """
    if isinstance(data_df, list):
        result_df = {}
        for data_d in data_df:
            result = evaluate(data_d, U, V)
            for k, v in result.items():
                if k in result_df:
                    result_df[k] += v / len(data_df)
                else:
                    result_df[k] = v / len(data_df)
        return result_df
    else:
        users = data_df[DEFAULT_USER_COL].to_numpy()
        items = data_df[DEFAULT_ITEM_COL].to_numpy()
        pred = np.array([np.dot(U[u], V[v]) for (u, v) in zip(users, items)])
        pred_df = pd.DataFrame(
            {
                DEFAULT_USER_COL: users,
                DEFAULT_ITEM_COL: items,
                DEFAULT_PREDICTION_COL: pred,
            }
        )

        result_dic = {}
        k_li = [5, 10, 20]
        metrics = [map_at_k, ndcg_at_k, precision_at_k, recall_at_k]
        if type(k_li) != list:
            k_li = [k_li]
        for k in k_li:
            for metric in metrics:
                result = metric(data_df, pred_df, k=k)
                result_dic[f"{metric.__name__}@{k}"] = result
        return result_dic
