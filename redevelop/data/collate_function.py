# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

from .batch_converter import BatchConverter

def LofD_to_DofL(raw_batch):
    """
    list of dict to dict of list
    :param raw_batch:
    :return:
    """
    batch_size = len(raw_batch)
    example = raw_batch[0]
    new_batch = {}
    for key in example.keys():
        new_batch[key] = []
        for i in range(batch_size):
            new_batch[key].append(raw_batch[i][key])
    return new_batch

def build_collate_fn(alphabet, data_type):
    batch_converter = BatchConverter(alphabet, data_type)
    def collate_fn(batch):
        if len(batch[0]) == 1:
            data = zip(*batch)
            data = LofD_to_DofL(data)
            data, anns = batch_converter(data)
            anns = None
        elif len(batch[0]) == 2:
            data, anns = zip(*batch)
            data = LofD_to_DofL(data)
            anns = LofD_to_DofL(anns)
            data, anns = batch_converter(data, anns)
        else:
            raise Exception("Unexpected Num of Components in a Batch")

        return data, anns

    return collate_fn



