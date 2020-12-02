#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys

sys.path.append("../")

import numpy as np
import pandas as pd
from numpy import linalg as LA
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

from src.utils import data_util
from src.utils.common_util import print_dict_as_table, save_to_csv
from utils.metric import evaluate, get_mse
from utils.optim import *
from utils.preprocessing import generate_train_data

"""
In this sample case, we get rid of data preprocessing step and we only give optimization steps, for movielens dataset, user owns two level hierarchical structure, and item owns two level hierarchical structure, so we have n:number of users; m: number of items; n1:group number of users; m1: group number of items.

After data preprocessing, we have already have X(user flat feature), Y(item flat feature), P_1(user hierarchy), Q_1(item hierarchy)

z: dimension of user flat feature

d: hidden vector dimension for matrix factorization

q: dimension of item flat feature
"""


def initialize(z, d, q):
    S1 = np.random.rand(z, d)
    S2 = np.random.rand(q, d)
    W1 = np.random.rand(z, z)
    W2 = np.random.rand(q, q)
    return S1, S2, W1, W2


class HIRE:
    def __init__(
        self,
        user_flat_feature,
        item_flat_feature,
        user_hierarchy=None,
        item_hierarchy=None,
        gamma=0.5,
        theta=0.5,
        corrupted_rate=0.2,
        beta=0.5,
        lamda=1,
        alpha=0.5,
        d_hidden=32,
    ):
        # user flat features
        self.u_flat = user_flat_feature.T
        self.i_flat = item_flat_feature.T
        self.u_hier = user_hierarchy
        self.i_hier = item_hierarchy
        (
            self.gamma,
            self.theta,
            self.corrupted,
            self.beta,
            self.lamda,
            self.alpha,
            self.d,
        ) = (gamma, theta, corrupted_rate, beta, lamda, alpha, d_hidden)
        self.z = self.u_flat.shape[0]
        self.q = self.i_flat.shape[0]
        # self.n1 = user_hierarchy.shape[0]
        # self.n = user_hierarchy.shape[1]
        # self.m1 = item_hierarchy.shape[1]
        # self.m = item_hierarchy.shape[0]

    def train(self, train_data, valid_df, test_df, optim_steps=1000, verbose=True):
        # initialize
        S1, S2, W1, W2 = initialize(self.z, self.d, self.q)

        train_loss_record = []
        test_loss_record = []

        model_nmf = NMF(n_components=self.d, init="random")
        U = model_nmf.fit_transform(train_data)
        V = model_nmf.components_

        # model_2_nmf = NMF(n_components=self.m1, init="random")
        # V2 = model_2_nmf.fit_transform(V)
        # V1 = model_2_nmf.components_
        #
        # model_3_nmf = NMF(n_components=self.n1, init="random")
        # U1 = model_3_nmf.fit_transform(U)
        # U2 = model_3_nmf.components_

        # if loss increase for five future steps, stop
        stop_time = 0
        times, old_loss = 0, 100000
        # V = V.T
        for _ in range(optim_steps):
            # U = U1.dot(U2)
            # V = V2.dot(V1)
            # pred = U.dot(V)
            # loss = get_mse(pred, train_data)
            # # loss_test = get_mse(pred, test_data)
            # if loss > old_loss:
            #     if stop_time == 5:
            #         break
            #     else:
            #         stop_time += 1
            # else:
            #     stop_time = 0

            # old_loss = loss

            if verbose:
                if times % 100 == 0:
                    valid_result = evaluate(valid_df, U, V.T)
                    print_dict_as_table(valid_result, tag="valid_result")
                    test_result = evaluate(test_df, U, V.T)
                    print_dict_as_table(test_result, tag="test_result")
                    # print(
                    #     "[Info] At time-step {}, test data mse loss is {}".format(
                    #         times, valid_result
                    #     ),
                    # )
            W1 -= Lipschitz_W1(
                self.u_flat, self.corrupted, self.gamma, self.z
            ) * SGD_W1(self.u_flat, self.corrupted, self.gamma, S1, U, self.z, W1)
            W2 -= Lipschitz_W2(
                self.i_flat, self.corrupted, self.theta, self.q
            ) * SGD_W2(V, self.corrupted, self.theta, S2, self.i_flat, W2, self.q)
            S1 -= Lipschitz_S1(U) * SGD_S1(W1, self.u_flat, U, self.gamma, S1)
            S2 -= Lipschitz_S2(V) * SGD_S2(W2, self.i_flat, V, S2, self.theta)
            times += 1

            # train_loss_record.append(loss)
            # test_loss_record.append(loss_test)
        return test_result


# a test demo
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(__file__))

    # user_flat_feature = np.loadtxt(os.path.join(base_path, "data", "X.txt"))
    # item_flat_feature = np.loadtxt(os.path.join(base_path, "data", "Y.txt"))
    # user_hierarchy = np.loadtxt(os.path.join(base_path, "data", "user_hierarchy.txt"))
    # item_hierarchy = np.loadtxt(os.path.join(base_path, "data", "item_hierarchy.txt"))
    # we do five fold cross-validation here
    # test_loss = []
    # you can grid search gamma, corrupted_rate, beta, lamda,alpha,d_hidden over here, for simplicity,  I set all values equal to 0.5, but these are definitely not the best hyper-parameters.

    datasets = ["dunnhumby", "tafeng", "instacart_25", "instacart"]
    for emb_dim in [64]:
        for dataset in datasets:
            config = {"dataset": dataset, "data_split": "temporal_basket", "n_neg": 100}
            data = data_util.Dataset(config)
            data.load_user_item_fea()
            data.init_train_items()
            HIRE_model = HIRE(data.user_feat, data.item_feat)
            test_result = HIRE_model.train(
                data.R, data.valid[0], data.test, optim_steps=1000
            )
            test_result["dataset"] = dataset
            test_result["model"] = "HIRE"
            test_result["emb_dim"] = emb_dim
            save_to_csv(test_result, "HIRE_result.csv")
