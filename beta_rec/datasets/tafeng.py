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

# download_url
TAFENG_URL = r"http://www.bigdatalab.ac.cn/benchmark/upload/download_source/f2ce090f-5642-8eb5-358c-4b58a2aba231_data.zip"

# processed data url
TAFENG_LEAVE_ONE_OUT_URL = r"https://1drv.ms/u/s!AjMahLyQeZqugWw1iWQHgI2NNbuM?e=LwEbEc"
TAFENG_RANDOM_SPLIT_URL = r"https://1drv.ms/u/s!AjMahLyQeZqugWbXQ__YWqF9v_7x?e=NjX5VQ"
TAFENG_TEMPORAL_SPLIT_URL = r"https://1drv.ms/u/s!AjMahLyQeZqugWp1Y1JefMXZr0ng?e=OoAgwD"


class Tafeng(DatasetBase):
    def __init__(self):
        """Tafeng

        Tafeng dataset.
        The dataset can not be download by the url,
        you need to down the dataset by 'https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset/download'
        then put it into the directory `tafeng/raw`
        """
        super().__init__(
            "tafeng",
            manual_download_url=TAFENG_URL,
            processed_random_split_url=TAFENG_RANDOM_SPLIT_URL,
        )

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = os.path.join(self.raw_path, "train.txt")
        if not os.path.exists(file_name):
            self.download()

        original_train_file = os.path.join(self.raw_path, "train.txt")
        original_test_file = os.path.join(self.raw_path, "test.txt")
        # initial dataframe
        interaction_list = []
        with open(original_train_file) as ori_test_df:
            for line in ori_test_df:
                temp_list = line.replace("\n", "\t").split("\t")
                # replace '\n' in the end of the line by '\t'
                # split line by '\t'
                # store splited items in a list
                order_id = temp_list[0]
                item_ids_list = temp_list[1:-3]  # itemids
                time_order = temp_list[-2].replace("-", "")
                user_id = temp_list[-3]
                for item_id in item_ids_list:
                    interaction_list.append(
                        [order_id, user_id, item_id, "train", "1", time_order]
                    )
            print(len(interaction_list))
        with open(original_test_file) as ori_test_df:
            for line in ori_test_df:
                temp_list = line.replace("\n", "\t").split("\t")
                # replace '\n' in the end of the line by '\t'
                # split line by '\t'
                # store splited items in a list
                order_id = temp_list[0]
                item_ids_list = temp_list[1:-3]  # itemids
                time_order = temp_list[-2].replace("-", "")
                user_id = temp_list[-3]
                for item_id in item_ids_list:
                    interaction_list.append(
                        [order_id, user_id, item_id, "train", "1", time_order]
                    )
            print(len(interaction_list))
        interactions = np.array(interaction_list)
        
        full_data = pd.DataFrame(
            data={
                DEFAULT_ORDER_COL: interactions[:,0],
                DEFAULT_USER_COL: interactions[:,1],
                DEFAULT_ITEM_COL: interactions[:,2],
                DEFAULT_FLAG_COL: interactions[:,3],
                DEFAULT_RATING_COL: interactions[:,4],
                DEFAULT_TIMESTAMP_COL: interactions[:,5],
            }
        )
        self.save_dataframe_as_npz(
            full_data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )