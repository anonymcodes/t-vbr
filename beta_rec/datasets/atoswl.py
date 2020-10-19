import os
import pandas as pd
import numpy as np

from beta_rec.utils.constants import (
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_ORDER_COL,
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_FLAG_COL,
    DEFAULT_RATING_COL,
)
from beta_rec.datasets.dataset_base import DatasetBase


class AtosWl(DatasetBase):
    def __init__(self):
        """ATOS Worldline dataset

        ATOS Worldline dataset.
        The dataset can not be download by the url,
        you need to down the dataset by 'https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset/download'
        then put it into the directory `atoswl/raw`
        """
        super().__init__(
            "atoswl",
            manual_download_url="",
            processed_random_split_url="",
        )

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        original_file = os.path.join(self.raw_path, "WORLDLINE_ANONYMIZED_DATASET_v2.csv")
        data = pd.read_csv(original_file)

        item_ids = data["ID_PRODUCT"].unique()
        len(item_ids)
        item_dic = {}
        item_id = 0
        for item in item_ids:
            item_dic[item] = item_id
            item_id += 1

        user_ids = data["ID_CLIENTE"].unique()
        len(user_ids)
        user_dic = {}
        user_id = 0
        for user in user_ids:
            user_dic[user] = user_id
            user_id += 1

        data[DEFAULT_USER_COL] = data["ID_CLIENTE"].apply(lambda x: user_dic[x])
        data[DEFAULT_ITEM_COL] = data["ID_PRODUCT"].apply(lambda x: item_dic[x])
        data[DEFAULT_TIMESTAMP_COL] = data["DIA"].apply(lambda x: int(x.replace("-", "")))
        data[DEFAULT_ORDER_COL] = data["NUM_TICKET"]
        data[DEFAULT_RATING_COL] = 1
        full_data = data[[DEFAULT_USER_COL,DEFAULT_ORDER_COL,DEFAULT_ITEM_COL,DEFAULT_RATING_COL,DEFAULT_TIMESTAMP_COL]]
        self.save_dataframe_as_npz(
            full_data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )
