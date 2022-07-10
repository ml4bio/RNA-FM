# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com

CATH 20201021 database
"""

import os
import pandas as pd
from torch.utils.data import Dataset

from .utils import *

class BIO_SEQ(Dataset):
    def __init__(self, root, theme, data_type="seq", set_name="train", transform=None,
                 length_limit=[1, 1023], length_subsample_mode="abandon",
                 depth_limit=[1, 32], depth_subsample_mode="random", use_cache=2):
        """
        :param root: root path of dataset - however not all of stuffs under this root path
        :param theme: "protein", "rna", "dna"
        :param data_type: seq, msa, cov, seq+msa, seq+cov ...
        :param label_type: 1d, 2d
        :param set_name: "train", "valid", "test"
        :param transform: useless so far
        :param length_limit: closed interval [] for length limit, where -1 indicates infinity
        :param depth_limit: closed interval [] for depth limit, where -1 indicates infinity
        """
        self.root = root
        self.theme = theme
        self.data_type = data_type
        self.set_name = set_name
        self.transform = transform
        self.length_limit = length_limit
        self.length_subsample_mode = length_subsample_mode
        self.depth_limit = depth_limit
        self.depth_subsample_mode = depth_subsample_mode

        if self.data_type == "seq" and self.depth_limit != [-1, -1]:
            raise Exception("With Pure Sequence Input, the Depth should be 1")

        self.use_cache = use_cache

        # 1. Create Paths
        # (1) data
        self.data_dir = {}

        # Pure Sequence
        #self.data_dir["seq"] = {"dir": os.path.join(self.root, self.set_name, "seq"), "ext":".seq"}
        # Multiple Sequence Alignment
        #self.data_dir["msa"] = {"dir": os.path.join(self.root, self.set_name, "msa"), "ext":".a3m"}
        #if set_name == "valid" or set_name == "test":
        #    self.msa_dir = "/share/liyu/cath35_20201021/msa_downsample/fixed-random_{}".format(self.depth)

        # (2) anns
        self.anns_dir = {}
        self.classes = {}
        self.class_proportion = {}
        self.balance_weight = {}

        # RNA secondary structure
        #self.anns_dir["r-ss"] = {"dir": os.path.join(self.root, self.set_name, "cm"), "ext": ".npy"}
        #self.classes["r-ss"] = ["0", "1"]
        #self.class_proportion["r-ss"] = [0.99875, 0.00125] #[0.9975, 0.0025]
        #self.balance_weight["r-ss"] = [1/(p*len(self.class_proportion["r-ss"])) for p in self.class_proportion["r-ss"]]

        # (3) metadata
        self.metadata_path = None #os.path.join(self.root, "ann", set_name + ".csv")

        # (4) cache
        self.cache_dir = None #os.path.join(self.root, "cache")

        # 2. Create Data INFO
        # self.data, self.anns, self.stats = self.__dataset_info(self.metadata_path, self.data_dir, self.anns_dir,
        #                                                       self.length_limit, self.depth_limit, self.set_name)

        # 3. Create Data Reader
        #self.data_reader = DataReader(self.data_type, use_cache=self.use_cache, cache_dir=self.cache_dir)

    @classmethod
    def create_dataset_with_name(cls, dataset_name, set_name, root_path, transforms):
        # for example, "spot-rna-bprna_seq_L:1,1023_D:-1,-1"
        _, data_type, length_limit, depth_limit = dataset_name.split("_")
        if "L:" in length_limit:
            length_limit = length_limit.replace("L:", "").replace("[", "").replace("]", "").split(",")
            length_limit = [int(limit) for limit in length_limit]
        else:
            raise Exception("Wrong Typos")
        if "D:" in depth_limit:
            depth_limit = depth_limit.replace("D:", "").replace("[", "").replace("]", "").split(",")
            depth_limit = [int(limit) for limit in depth_limit]
        else:
            raise Exception("Wrong Typos")
        dataset = cls(root=root_path, set_name=set_name, data_type=data_type, transform=transforms,
                      length_limit=length_limit, depth_limit=depth_limit)
        return dataset

    def __getitem__(self, index):
        data = self.data_reader.read_data(self.data[index])
        anns = self.data_reader.read_anns(self.anns[index])
        return data, anns

    def __len__(self):
        return len(self.data)   # self.stats.shape[0]

    def dataset_info(self, metadata_path, data_dir, anns_dir, len_limit, dep_limit, set_name):
        """
        :param name_path: txt record name list for specific set_name
        :param data_dir:
        :param msa_dir:
        :param ann1d_dir:
        :param ann2d_dir:
        :return:
        """
        data = []
        anns = []

        selected_indices = []

        src_df = pd.read_csv(metadata_path)
        for index, row in src_df.iterrows():
            name = row["filename"]
            #extension = row["extension"]
            length = int(row["length"])
            depth = int(row["depth"]) if "depth" in row else 1

            # check length and depth
            if (length < len_limit[0] and len_limit[0] != -1) or (length > len_limit[1] and len_limit[1] != -1):
                continue

            if (depth < dep_limit[0] and dep_limit[0] != -1) or (depth > dep_limit[1] and dep_limit[1] != -1):
                continue

            # check the integrality of files
            check_integrality = 1
            temp_data = {}
            for key in data_dir.keys():
                path = os.path.join(data_dir[key]["dir"], name + data_dir[key]["ext"])
                temp_data[key] = path
                if os.path.exists(path) != True:
                    print("{} doesn't exist.".format("{}: {}".format(key, path)))
                    check_integrality = 0
                    continue
            temp_anns = {}
            for key in anns_dir.keys():
                path = os.path.join(anns_dir[key]["dir"], name + anns_dir[key]["ext"])
                temp_anns[key] = path
                if os.path.exists(path) != True:
                    print("{} doesn't exist.".format("{}: {}".format(key, path)))
                    check_integrality = 0
                    continue
            if check_integrality == 0:
                continue

            # save paths in the list
            data.append(temp_data)
            anns.append(temp_anns)

            selected_indices.append(index)

            # truncation for debug
            # if len(selected_indices) >= 100:
            #    break

        print("{} Dataset Info:".format(set_name))
        print("Length-Frequency Table")
        selected_df = src_df.iloc[selected_indices]
        print(selected_df["length"].describe())  # value_counts())
        print("Depth-Frequency Table")
        selected_df = src_df.iloc[selected_indices]
        print(selected_df["depth"].describe())   # value_counts())

        return data, anns, selected_df

    # reset max_length
    def reset_length_limit(self, length_limit):
        self.length_limit = length_limit
        self.data, self.anns, self.stats = self.dataset_info(self.metadata_path, self.data_dir, self.anns_dir,
                                                             self.length_limit, self.depth_limit, self.set_name)


    def create_balance_weight(self, class_proportion):
        """
        the same as sklearn.utils.class_weight.compute_class_weight
        :param class_proportion: the sum should be 1
        :return:
        """
        balance_weight = [1 / (p * len(class_proportion)) for p in class_proportion]
        return balance_weight