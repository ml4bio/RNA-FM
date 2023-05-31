# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com

CATH 20201021 database
"""

import os
import pandas as pd
from redevelop.data.datasets.utils import *
from redevelop.data.datasets.bio_seq import BIO_SEQ

from Bio import SeqIO

class Custom(BIO_SEQ):
    def __init__(self, root, data_type="seq", set_name="train", transform=None,
                 length_limit=[1, 1023], length_subsample_mode="abandon",
                 depth_limit=[1, 32], depth_subsample_mode="random", use_cache=2):
        """
        :param root: root path of dataset - CATH. however not all of stuffs under this root path
        :param data_type: seq, msa
        :param label_type: 1d, 2d
        :param set_name: "train", "valid", "test"
        :param transform: useless so far
        :param length_limit: closed interval [] for length limit, where -1 indicates infinity
        :param depth_limit: closed interval [] for depth limit, where -1 indicates infinity
        """
        super().__init__(root=root, theme="rna", data_type=data_type, set_name=set_name, transform=transform,
                         length_limit=length_limit, length_subsample_mode=length_subsample_mode,
                         depth_limit=depth_limit, depth_subsample_mode=depth_subsample_mode, use_cache=use_cache)
        if os.path.exists(self.root) != True:
            raise Exception("'{}' does not exist!".format(self.root))

        if os.path.isdir(self.root):
            self.inputs_format = "folder"

            self.data_reader = DataReader(self.theme, self.data_type, use_cache=0)
            self.data = []

            filename_list = os.listdir(self.root)
            for filename in filename_list:
                file = os.path.join(self.root, filename)
                data = self.data_reader.read_data({"seq": file})
                self.data.append(data)

        elif os.path.isfile(self.root):
            self.inputs_format = "file"
            data_path = self.root
            record_iter = SeqIO.parse(data_path, "fasta")
            self.stats = []
            self.data = []
            rna_types = {}
            num_count = 0
            for index, record in enumerate(record_iter):
                formatted_seq = str(record.seq).upper().replace('T', "U").replace('~', "-")
                length = len(formatted_seq)
                depth = 1
                # check length and depth
                if (length < self.length_limit[0] and self.length_limit[0] != -1) or (length > self.length_limit[1] and self.length_limit[1] != -1):
                    continue

                if (depth < self.depth_limit[0] and self.depth_limit[0] != -1) or (depth > self.depth_limit[1] and self.depth_limit[1] != -1):
                    continue

                """
                rna_type = record.description.split(" ")[1]
                if len(rna_types.keys()) == 5 and rna_type not in rna_types:
                    continue
                if rna_type in rna_types:
                    if rna_types[rna_type] >= 10:
                        continue
                    else:
                        rna_types[rna_type] += 1
                else:
                    rna_types[rna_type] = 1

                if index > 20000:
                    print(rna_types.keys())
                    print(num_count)
                    break
                #"""

                self.data.append({"seq":(record.id, formatted_seq)})   #description

                """
                num_count += 1
                if num_count == 10 * 5:
                    print(rna_types.keys())
                    print(num_count)
                    break
                #"""
            print("Sequence Num: {}".format(len(self.data)))


    def __getitem__(self, index):
        data = self.data[index]
        anns = {"label": 1}

        return data, anns

    def __len__(self):
        return len(self.data)  # self.stats.shape[0]

