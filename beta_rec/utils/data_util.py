import os
from time import time
import numpy as np
import pandas as pd
import random
from tabulate import tabulate
import scipy.sparse as sp
from beta_rec.utils.common_util import get_random_rep, ensureDir
from beta_rec.utils.aliasTable import AliasTable
from beta_rec.utils.triple_sampler import Sampler
from beta_rec.datasets.data_load import (
    load_split_dataset,
    load_item_fea_dic,
    load_user_item_feature,
)
from beta_rec.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
)


pd.options.mode.chained_assignment = None  # default='warn'


def intersect_train_test(train, test):
    """ Get the intersect lists of users and items that exist in both train and test

    Args:
        train (DataFrame):
        test (DataFrame):

    Returns:
        users (list): users list
        items (list): items list

    """
    users = list(
        set(train[DEFAULT_USER_COL].unique().flatten()).intersection(
            set(test[DEFAULT_USER_COL].unique().flatten())
        )
    )
    items = list(
        set(train[DEFAULT_ITEM_COL].unique().flatten()).intersection(
            set(test[DEFAULT_ITEM_COL].unique().flatten())
        )
    )
    return users, items


def get_feat_dic(fea_array):
    fea_dic = {}
    for row in fea_array:
        fea_dic[row[0]] = row[1:]
    return fea_dic


def calc_sim(A):
    """Fastest way to calculate the cosine similarity
    see:
    https://stackoverflow.com/questions/17627219/
    """
    similarity = np.dot(A, A.T)

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    return cosine.T * inv_mag


def normalized_adj_single(adj):
    """ Missing docs

    Args:
        adj:

    Returns:

    """
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    # norm_adj = adj.dot(d_mat_inv)
    print("generate single-normalized adjacency matrix.")
    return norm_adj.tocoo()


def get_D_inv(adj):
    """ Missing docs

    Args:
        adj:

    Returns:

    """
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv


def check_adj_if_equal(adj):
    """ Missing docs
    
    Args:
        adj:

    Returns:

    """
    dense_A = np.array(adj.todense())
    degree = np.sum(dense_A, axis=1, keepdims=False)
    temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
    print("check normalized adjacency matrix whether equal to this laplacian matrix.")
    return temp


class Dataset(object):
    """
        Base Dataset class for all the model
    """

    def __init__(self, config):
        """Constructor

        Args:
            config:
        """
        self.config = config
        self.n_users = 0
        self.n_items = 0
        self.sub_set = 0
        self.random_dim = 512
        # subset of the dataset. use a small set of users and items if >0, otherwise use full dataset
        if "sub_set" in config:
            self.sub_set = config["sub_set"]
        if "random_dim" in config:
            self.random_dim = config["random_dim"]
        # data preprocessing for training and test data
        # To be replaced with new data method
        train, valid, test = load_split_dataset(config)
        self.train = self._intersect_train_test(train, test[0])
        self.n_train = len(self.train.index)
        self.valid = self._reindex_list(valid)
        self.test = self._reindex_list(test)
        self.item_sampler = AliasTable(
            self.train[DEFAULT_ITEM_COL].value_counts().to_dict()
        )
        self.user_sampler = AliasTable(
            self.train[DEFAULT_USER_COL].value_counts().to_dict()
        )
        self.init_item_fea()
        self.init_user_fea()

    def sample_triple_time(self, dump=True, load_save=False):
        """
        Sample triples or load triples samples from files. Only applicable for basket based Recommender
        Returns:
            None

        """
        sample_file_name = (
            "triple_"
            + self.config["dataset"]
            + (
                ("_" + str(self.config["percent"] * 100))
                if "percent" in self.config
                else ""
            )
            + (
                ("_" + str(self.config["time_step"]))
                if "time_step" in self.config
                else "_10"
            )
            + "_"
            + str(self.config["n_sample"])
            if "percent" in self.config
            else "" + ".csv"
        )
        self.process_path = os.path.join(
            self.config["root_dir"], self.config["process_dir"]
        )
        ensureDir(self.process_path)
        sample_file = os.path.join(self.process_path, sample_file_name)
        my_sampler = Sampler(
            self.train,
            sample_file,
            self.config["n_sample"],
            dump=dump,
            load_save=load_save,
        )
        return my_sampler.sample_by_time(self.config["time_step"])

    def sample_triple(self, dump=True, load_save=False):
        """
        Sample triples or load triples samples from files. Only applicable for basket based Recommender
        Returns:
            None

        """
        sample_file_name = (
            "triple_"
            + self.config["dataset"]
            + (
                ("_" + str(self.config["percent"] * 100))
                if "percent" in self.config
                else ""
            )
            + "_"
            + str(self.config["n_sample"])
            if "percent" in self.config
            else "" + ".csv"
        )
        self.process_path = os.path.join(
            self.config["root_dir"], self.config["process_dir"]
        )
        ensureDir(self.process_path)
        sample_file = os.path.join(self.process_path, sample_file_name)
        my_sampler = Sampler(
            self.train,
            sample_file,
            self.config["n_sample"],
            dump=dump,
            load_save=load_save,
        )
        return my_sampler.sample()

    def generate_train_data(self):
        """ Generate a rating matrix for interactions

        Returns:
            (sigma_matrix, rating_matrix)
            sigma_matrix with rating being 1
            rating_matrix with rating being the real rating
        """
        train_data = (
            self.train.groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL])
            .sum()
            .to_frame(DEFAULT_RATING_COL)
            .reset_index()
        )
        n = self.n_users
        m = self.n_items
        user_record = train_data.col_user.tolist()
        movie_record = train_data.col_item.tolist()
        ratings_record = train_data.col_rating.tolist()
        rating_matrix = np.zeros([n, m])
        sigma_matrix = np.zeros([n, m])
        for i in range(len(user_record)):
            rating_matrix[user_record[i] - 1, movie_record[i] - 1] = ratings_record[i]
            sigma_matrix[user_record[i] - 1, movie_record[i] - 1] = 1
        # load test_data
        print("data load finish")
        return sigma_matrix, rating_matrix

    def generate_sparse_train_data(self):
        """Generate a sparse matrix for interactions.

        Returns:
            coo_matrix
        """

        train_data = (
            self.train.groupby(["col_user", "col_item"])
            .sum()
            .to_frame("col_rating")
            .reset_index()
        )
        user_record = train_data.col_user.tolist()
        movie_record = train_data.col_item.tolist()
        ratings_record = train_data.col_rating.tolist()
        print("data load finish")
        return sp.coo_matrix((ratings_record, (user_record, movie_record)))

    def _intersect_train_test(self, train, test, implicit=True):
        """ process the dataset to reindex userID and itemID, also set rating as implicit feedback

        Parameters:
            train (pandas.DataFrame): training data with at least columns (col_user, col_item, col_rating) 
            test (pandas.DataFrame): test data with at least columns (col_user, col_item, col_rating)
                    test can be None, if so, we only process the training data
            implicit (bool): if true, set rating>0 to rating = 1 

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed.
        
        """
        users_ser, items_ser = intersect_train_test(train, test)
        self.n_users = -1
        self.n_items = -1
        while self.n_users != len(users_ser) or self.n_items != len(items_ser):
            test = test[
                test[DEFAULT_USER_COL].isin(users_ser)
                & test[DEFAULT_ITEM_COL].isin(items_ser)
            ]
            train = train[
                train[DEFAULT_USER_COL].isin(users_ser)
                & train[DEFAULT_ITEM_COL].isin(items_ser)
            ]
            self.n_users = train[DEFAULT_USER_COL].nunique()
            self.n_items = train[DEFAULT_ITEM_COL].nunique()
            users_ser, items_ser = intersect_train_test(train, test)
            print(f"After intersection, train statistics")
            print(
                tabulate(
                    train.agg(["count", "nunique"]),
                    headers=test.columns,
                    tablefmt="psql",
                    disable_numparse=True,
                )
            )
            print(f"After intersection, test statistics")
            print(
                tabulate(
                    test.agg(["count", "nunique"]),
                    headers=test.columns,
                    tablefmt="psql",
                    disable_numparse=True,
                )
            )
        self.user_pool = users_ser
        self.item_pool = items_ser

        # Reindex user and item index
        self.user2id = dict(zip(np.array(self.user_pool), np.arange(self.n_users)))
        self.id2user = {self.user2id[k]: k for k in self.user2id}

        self.item2id = dict(zip(np.array(self.item_pool), np.arange(self.n_items)))
        self.id2item = {self.item2id[k]: k for k in self.item2id}

        if self.sub_set > 0:  # for small dataset
            print("using a subset of the dataset:", self.sub_set)
            self.user_pool = self.user_pool[: self.sub_set]
            self.item_pool = self.item_pool[: self.sub_set]
        return self._reindex(train, implicit)

    def _reindex_list(self, df_list):
        """_reindex for list of dataset. For example, validate and test can be a list for evaluation
        
        """
        df_list_new = []
        for df in df_list:
            df = df[
                df[DEFAULT_USER_COL].isin(self.user_pool)
                & df[DEFAULT_ITEM_COL].isin(self.item_pool)
            ]
            df_list_new.append(self._reindex(df))
        return df_list_new

    def _reindex(self, df, implicit=True):
        """ 
        Process dataset to reindex userID and itemID, also set rating as implicit feedback

        Parameters:
            df (pandas.DataFrame): dataframe with at least columns (col_user, col_item, col_rating) 
            implicit (bool): if true, set rating>0 to rating = 1 

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed.
        
        """

        # If testing dataset is None
        if df is None:
            return None

        # Map user_idx and item_idx
        df[DEFAULT_USER_COL] = df[DEFAULT_USER_COL].apply(lambda x: self.user2id[x])
        df[DEFAULT_ITEM_COL] = df[DEFAULT_ITEM_COL].apply(lambda x: self.item2id[x])

        # If implicit feedback, set rating as 1.0 or 0.0
        if implicit:
            df[DEFAULT_RATING_COL] = df[DEFAULT_RATING_COL].apply(
                lambda x: float(x > 0)
            )
        return df

    def init_item_fea(self):
        """
        initialize item feature

        """
        config = self.config
        if "item_fea_type" in config:
            fea_type = config["item_fea_type"]
        else:
            fea_type = "random"
        data_str = config["dataset"]
        print(
            "Loading item feature for dataset:", data_str, " type:", fea_type,
        )

        if fea_type == "random":
            self.item_feature = get_random_rep(self.n_items, self.random_dim)
        elif fea_type == "one_hot":
            item_fea_dic = load_item_fea_dic(config, fea_type="one_hot")
            self.item_feature = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
        elif fea_type == "word2vec":
            item_fea_dic = load_item_fea_dic(config, fea_type="word2vec")
            self.item_feature = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
        elif fea_type == "bert":
            item_fea_dic = load_item_fea_dic(config, fea_type="bert")
            self.item_feature = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
        elif fea_type == "random_one_hot" or fea_type == "one_hot_random":
            rand_item_fea = get_random_rep(self.n_items, 512)
            item_fea_dic = load_item_fea_dic(config, fea_type="one_hot")
            item_fea_list = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate((item_fea_list, rand_item_fea), axis=1)
        elif fea_type == "random_cate" or fea_type == "cate_random":
            rand_item_fea = get_random_rep(self.n_items, 512)
            item_fea_dic = load_item_fea_dic(config, fea_type="cate")
            item_fea_list = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate((item_fea_list, rand_item_fea), axis=1)
        elif fea_type == "random_word2vec" or fea_type == "word2vec_random":
            rand_item_fea = get_random_rep(self.n_items, 512)
            item_fea_dic = load_item_fea_dic(config, fea_type="word2vec")
            item_fea_list = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate((item_fea_list, rand_item_fea), axis=1)
        elif fea_type == "random_bert" or fea_type == "bert_random":
            rand_item_fea = get_random_rep(self.n_items, 512)
            item_fea_dic = load_item_fea_dic(config, fea_type="bert")
            item_fea_list = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate((item_fea_list, rand_item_fea), axis=1)
        elif fea_type == "word2vec_one_hot" or fea_type == "one_hot_word2vec":
            item_fea_dic1 = load_item_fea_dic(config, fea_type="one_hot")
            item_fea_dic2 = load_item_fea_dic(config, fea_type="word2vec")
            item_fea_list1 = np.array(
                [item_fea_dic1[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            item_fea_list2 = np.array(
                [item_fea_dic2[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate((item_fea_list1, item_fea_list2), axis=1)
        elif (
            fea_type == "word2vec_one_hot_random"
            or fea_type == "random_one_hot_word2vec"
            or fea_type == "one_hot_random_word2vec"
            or fea_type == "random_word2vec_one_hot"
        ):
            rand_item_fea = get_random_rep(self.n_items, 512)
            item_fea_dic1 = load_item_fea_dic(config, fea_type="one_hot")
            item_fea_dic2 = load_item_fea_dic(config, fea_type="word2vec")
            item_fea_list1 = np.array(
                [item_fea_dic1[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            item_fea_list2 = np.array(
                [item_fea_dic2[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate(
                (item_fea_list1, item_fea_list2, rand_item_fea), axis=1
            )
        elif (
            fea_type == "word2vec_one_hot_bert"
            or fea_type == "bert_one_hot_word2vec"
            or fea_type == "one_hot_bert_word2vec"
        ):
            item_fea_dic1 = load_item_fea_dic(config, fea_type="one_hot")
            item_fea_dic2 = load_item_fea_dic(config, fea_type="word2vec")
            item_fea_dic3 = load_item_fea_dic(config, fea_type="bert")
            item_fea_list1 = np.array(
                [item_fea_dic1[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            item_fea_list2 = np.array(
                [item_fea_dic2[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            item_fea_list3 = np.array(
                [item_fea_dic3[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate(
                (item_fea_list1, item_fea_list2, item_fea_list3), axis=1
            )
        elif (
            fea_type == "random_word2vec_one_hot_bert"
            or fea_type == "bert_one_hot_word2vec_random"
            or fea_type == "one_hot_random_bert_word2vec"
            or fea_type == "one_hot_bert_random_word2vec"
            or fea_type == "one_hot_word2vec_bert_random"
            or fea_type == "random_one_hot_word2vec_bert"
        ):
            rand_item_fea = get_random_rep(self.n_items, 512)
            item_fea_dic1 = load_item_fea_dic(config, fea_type="one_hot")
            item_fea_dic2 = load_item_fea_dic(config, fea_type="word2vec")
            item_fea_dic3 = load_item_fea_dic(config, fea_type="bert")
            item_fea_list1 = np.array(
                [item_fea_dic1[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            item_fea_list2 = np.array(
                [item_fea_dic2[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            item_fea_list3 = np.array(
                [item_fea_dic3[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate(
                (item_fea_list1, item_fea_list2, item_fea_list3, rand_item_fea), axis=1
            )
        else:
            print(
                "[ERROR]: CANNOT support feature type {}! intialize with random feature".format(
                    fea_type
                )
            )
            self.item_feature = get_random_rep(self.n_items, self.random_dim)

    def init_user_fea(self):
        """
        initialize user feature
        VBCAR model

        """
        if "user_fea_type" in self.config:
            fea_type = self.config["user_fea_type"]
        else:
            fea_type = "random"

        print(
            "init user featrue for dataset:", self.config["dataset"], " type:", fea_type
        )
        if fea_type == "random":
            self.user_feature = get_random_rep(self.n_users, self.random_dim)
        else:
            print(
                "[ERROR]: CANNOT support other feature type, use 'random' user featrue instead!"
            )
            self.user_feature = get_random_rep(self.n_users, self.random_dim)
            # load_user_fea(config)

    def get_adj_mat(self):
        """ Get the adjacent matrix, if not previously stored then call the function to create
        This method is for NGCF model
        Return:
            Different types of adjacment matrix
        """
        self.init_train_items()

        process_file_name = (
            "ngcf_"
            + self.config["dataset"]
            + "_"
            + self.config["data_split"]
            + (
                ("_" + str(self.config["percent"] * 100))
                if "percent" in self.config
                else ""
            )
        )
        self.process_path = os.path.join(
            self.config["root_dir"], self.config["process_dir"]
        )
        process_file_name = os.path.join(self.process_path, process_file_name)
        ensureDir(process_file_name)
        print(process_file_name)
        try:
            t1 = time()
            adj_mat = sp.load_npz(os.path.join(process_file_name, "s_adj_mat.npz"))
            norm_adj_mat = sp.load_npz(
                os.path.join(process_file_name, "s_norm_adj_mat.npz")
            )
            mean_adj_mat = sp.load_npz(
                os.path.join(process_file_name, "s_mean_adj_mat.npz")
            )
            print("already load adj matrix", adj_mat.shape, time() - t1)
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(os.path.join(process_file_name, "s_adj_mat.npz"), adj_mat)
            sp.save_npz(
                os.path.join(process_file_name, "s_norm_adj_mat.npz"), norm_adj_mat
            )
            sp.save_npz(
                os.path.join(process_file_name, "s_mean_adj_mat.npz"), mean_adj_mat
            )
        return adj_mat, norm_adj_mat, mean_adj_mat

    def load_user_item_fea(self):
        """ Load user and item features from datasets.

        Returns:

        """
        print("Load user and item features from datasets")
        user_feat, item_feat = load_user_item_feature(self.config)
        user_feat_li = [None for i in range(self.n_users)]
        item_feat_li = [None for i in range(self.n_items)]
        for user in user_feat:
            if user[0] in self.user2id:
                user_feat_li[self.user2id[user[0]]] = user[1:]
        for item in item_feat:
            if item[0] in self.item2id:
                item_feat_li[self.item2id[item[0]]] = item[1:]
        self.user_feat = np.stack(user_feat_li)
        self.item_feat = np.stack(item_feat_li)

    def make_fea_sim_mat(self):
        """ Note that the first column is the user/item ID
        Returns
            normalized_adj_single
        """
        self.load_user_item_fea()
        self.init_train_items()
        user_sim_mat = sp.csr_matrix(calc_sim(self.user_feat))
        item_sim_mat = sp.csr_matrix(calc_sim(self.item_feat))
        return (
            normalized_adj_single(user_sim_mat + sp.eye(user_sim_mat.shape[0])),
            normalized_adj_single(item_sim_mat + sp.eye(item_sim_mat.shape[0])),
        )

    def create_adj_mat(self):
        """ Create adjacent matirx from the user-item interaction matrix
        """
        t1 = time()
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        adj_mat[: self.n_users, self.n_users :] = R
        adj_mat[self.n_users :, : self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print("already create adjacency matrix", adj_mat.shape, time() - t1)
        t2 = time()
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        print("already normalize adjacency matrix", time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [random.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print("refresh negative pools", time() - t1)

    def sample(self, batch_size):
        """ Sample users, their positive items and negative items

        Returns:
            users (list)
            pos_items (list)
            neg_items (list)
        """
        if batch_size <= self.n_users:
            users = random.sample(range(self.n_users), batch_size)
        else:
            users = [random.choice(range(self.n_users)) for _ in range(batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return random.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        return users, pos_items, neg_items

    def sample_all_users_pos_items(self):
        self.all_train_users = []

        self.all_train_pos_items = []
        for u in range(self.n_users):
            self.all_train_users += [u] * len(self.train_items[u])
            self.all_train_pos_items += self.train_items[u]

    def epoch_sample(self):
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        neg_items = []
        for u in self.all_train_users:
            neg_items += sample_neg_items_for_u(u, 1)

        perm = np.random.permutation(len(self.all_train_users))
        users = np.array(self.all_train_users)[perm]
        pos_items = np.array(self.all_train_pos_items)[perm]
        neg_items = np.array(neg_items)[perm]
        return users, pos_items, neg_items

    def init_train_items(self):
        self.n_train = 0
        self.train_items = {}
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        user_np = np.array(self.train[DEFAULT_USER_COL])
        item_np = np.array(self.train[DEFAULT_ITEM_COL])
        for u in range(self.n_users):
            index = list(np.where(user_np == u)[0])
            i = item_np[index]
            self.train_items[u] = i
            for item in i:
                self.R[u, item] = 1
                self.n_train += 1
